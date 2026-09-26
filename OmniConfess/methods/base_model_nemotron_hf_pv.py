# -*- coding: utf-8 -*-
"""
base_model_nemotron_hf.py — Single-pass greedy baseline on
NVIDIA Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16, transformers
trust_remote_code path.

Why not vLLM (the model card's primary path): on this host
  * PyPI vllm==0.20.0 wheels link libcudart.so.13 — driver 550 (CUDA 12.4)
    cannot run CUDA 13 binaries;
  * the official cu129 container needs ~15-20 GB on /, which is 100% full.
The repo ships full HF remote code (modeling.py NemotronH_Nano_Omni_Reasoning_V3
+ processing.py + Parakeet audio + C-RADIO vision) that targets transformers
5.x, so we run it in the dedicated /slow_share/tsq/venvs/nemotron-hf venv
(torch 2.6.0+cu124, transformers 5.6.2) — driver-compatible and GPFS-local.

Structural mirror of methods/base_model_qwen3_omni.py: same shared v1 prompts,
same data_loader, same result schema, same phi-Decoding style trajectory
(per-token logprobs via output_scores).

Media mapping:
  video  -> cv2 native-res frames @ fps=2 (cap 64) -> processor(videos=...)
            np [T,H,W,C]; avhbench audio == the video's own track, so the SAME
            mp4 also feeds audio= (16 kHz mono waveform) — both modalities in.
  audio  -> torchaudio.load (mp4/wav), fallback: static ffmpeg -> wav
  image  -> PIL RGB -> processor(images=...)
Prompt: tokenizer.apply_chat_template(messages, add_generation_prompt=True,
        enable_thinking=False) — placeholders <video>/<audio>/<image> are
        expanded by the processor into context tokens.

Lives on GPFS (repo / is 100% full): sys.path picks up the repo's
OmniConfess package via NEMO_REPO or the default below.

Usage:
  CUDA_VISIBLE_DEVICES=5 /slow_share/tsq/venvs/nemotron-hf/bin/python \
      base_model_nemotron_hf.py --datasets pubmedqa \
      --data_path $REPO/OmniConfess/OmniHalluBench/pubmedqa/pubmedqa.json \
      --output_dir ./outputs/pubmedqa_nemotron --dry_run
"""
import os, sys, time, json, argparse, random
import numpy as np
import torch
import cv2

REPO = os.environ.get(
    'NEMO_REPO', '/home/tsq/rhq/omniconfess/OmniConfess')
sys.path.insert(0, os.path.join(REPO, 'OmniConfess', 'methods'))
from shared import (
    load_dataset, load_media_for_item, resolve_media_path,
    get_system_prompt, get_user_prompt,
    make_result, save_result,
)
from shared.data_loader import load_image

DEFAULT_MODEL = '/slow_share/tsq/MODELS/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16'
AUDIO_SR = 16000          # config sound_config.sampling_rate
VIDEO_FPS = 2             # model-card guidance for <=720p video
VIDEO_MAX_FRAMES = 64     # our videos are ~10s -> ~20 frames



def _repair_buffers(model, model_path, device):
    """Fix transformers 5.x remote-code load artifacts on this checkpoint.

    Symptom (CPU repro): `IndexError: index 94090993518273 is out of bounds for
    dimension 0 with size 4` at radio_model.py:196 -- `summary_idxs` (an int64
    index buffer, deterministic = [0,1,2] from config teachers) is NOT in the
    safetensors and transformers' init pass leaves it as uninitialized garbage,
    which on CUDA surfaces as a device-side assert on EVERY image/video item
    (text-only items are unaffected).

    Additionally, 4 float buffers are registered non-persistent by the remote
    code ( conditioner norm_mean/norm_std, Whisper mel featurizer fb/window ),
    so the checkpoint copies are reported UNEXPECTED and never loaded; the
    constructor defaults are almost surely identical, but copying the
    checkpoint values in explicitly removes the silent-drift risk.
    """
    import json as _json
    from safetensors.torch import load_file as _load_file

    # 1) summary_idxs: rebuild deterministically from config teachers
    cfg = _json.load(open(os.path.join(model_path, 'config.json')))
    teachers = cfg['vision_config']['args']['teachers']
    idxs = [i for i, t in enumerate(teachers) if t.get('use_summary', True)]
    radio = model.vision_model.radio_model
    fixed = torch.tensor(idxs, dtype=torch.long, device=device)
    bad = (radio.summary_idxs is None or
           radio.summary_idxs.device != fixed.device or
           not torch.equal(radio.summary_idxs.long(), fixed))
    if bad:
        print(f'[repair] summary_idxs garbage -> {idxs}', flush=True)
        radio.summary_idxs = fixed

    # 2) copy non-persistent float constants from the checkpoint
    idx = _json.load(open(os.path.join(model_path, 'model.safetensors.index.json')))['weight_map']
    want = {
        'vision_model.radio_model.input_conditioner.norm_mean':
            ('vision_model', 'radio_model', 'input_conditioner', 'norm_mean'),
        'vision_model.radio_model.input_conditioner.norm_std':
            ('vision_model', 'radio_model', 'input_conditioner', 'norm_std'),
        'sound_encoder.encoder.feature_extractor.featurizer.fb':
            ('sound_encoder', 'encoder', 'feature_extractor', 'featurizer', 'fb'),
        'sound_encoder.encoder.feature_extractor.featurizer.window':
            ('sound_encoder', 'encoder', 'feature_extractor', 'featurizer', 'window'),
    }
    fixes = 0
    for key, attr_path in want.items():
        shard = idx.get(key)
        if shard is None:
            continue
        try:
            t = _load_file(os.path.join(model_path, shard))[key]
            node = model
            for a in attr_path[:-1]:
                node = getattr(node, a)
            cur = getattr(node, attr_path[-1], None)
            if cur is None or cur.shape != t.shape or not torch.allclose(cur.float().cpu(), t.float(), atol=1e-3):
                setattr(node, attr_path[-1], t.to(device=device, dtype=cur.dtype if cur is not None else t.dtype))
                fixes += 1
        except Exception as e:
            print(f'[repair] skip {key}: {type(e).__name__}: {e}', flush=True)
    print(f'[repair] done ({fixes} constant buffers restored, summary_idxs ok)', flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_video_frames_native(video_path, fps=VIDEO_FPS, max_frames=VIDEO_MAX_FRAMES):
    """Native-resolution RGB frames [T,H,W,C] sampled at `fps` (no resize: the
    processor's InternVL-style dynamic tiling handles resolution)."""
    if not video_path or not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    step = max(1, round(native_fps / fps))
    indices = list(range(0, total, step))[:max_frames]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        from PIL import Image
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    # the remote image_processor's video rule reads .width/.height -> PIL
    return frames if frames else None


def load_audio_wave(audio_path, sr=AUDIO_SR):
    """16 kHz mono float waveform from wav/mp4."""
    if not audio_path or not os.path.exists(audio_path):
        return None
    try:
        import torchaudio
        wav, orig_sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if orig_sr != sr:
            wav = torchaudio.functional.resample(wav, orig_sr, sr)
        out = wav.squeeze(0).numpy()
        return out if out.size else None
    except Exception as e:
        print(f'  [audio] torchaudio failed ({e}), trying ffmpeg->wav', flush=True)
    try:
        import subprocess, tempfile, shutil, soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
            tmp = tf.name
        ffmpeg = shutil.which('ffmpeg') or             '/home/tsq/rhq/omniconfess/bin/ffmpeg'   # bundled static build
        r = subprocess.run([ffmpeg, '-y', '-i', audio_path, '-ac', '1',
                            '-ar', str(sr), '-f', 'wav', tmp],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=120)
        if r.returncode == 0:
            wav, _ = sf.read(tmp, dtype='float32')
            os.unlink(tmp)
            return wav if wav.size else None
    except Exception as e:
        print(f'  [audio] ffmpeg fallback failed: {e}', flush=True)
    return None


def token_logprobs(gen_ids, scores):
    """Per-token logprob of each generated token (greedy scores)."""
    out = []
    for tid, logits in zip(gen_ids.tolist(), scores):
        logits = logits.float()
        if logits.dim() > 1:
            logits = logits[0]
        lp = torch.log_softmax(logits, dim=-1)[tid]
        out.append(round(lp.item(), 4))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',      type=str, required=True,
                        help='cmm | phd | haloquest | avhbench | ragtruth | pubmedqa')
    parser.add_argument('--data_path',     type=str, required=True)
    parser.add_argument('--data_root',     type=str, default=None,
                        help='defaults to the dataset dir of data_path')
    parser.add_argument('--output_dir',    type=str, default='./outputs')
    parser.add_argument('--file_name',     type=str, default='base_model_nemotron')
    parser.add_argument('--time_path',     type=str, default='./outputs/time')
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument('--max_new_tokens',type=int, default=512)
    parser.add_argument('--prompt_version',type=str, default='v1',
                        choices=['v0','v1','v2','v3','v4','v5','v6'])
    parser.add_argument('--model_path',    type=str,
                        default=os.environ.get('NEMOTRON_MODEL_PATH', DEFAULT_MODEL))
    parser.add_argument('--enable_thinking', action='store_true',
                        help='Keep the reasoning chain on (default OFF for '
                             'comparability with the other backbone arms)')
    parser.add_argument('--record_process', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--dry_run',       action='store_true',
                        help='Only process first 3 items (smoke test)')
    args = parser.parse_args()
    if not args.data_root:
        args.data_root = os.path.join(
            REPO, 'OmniHalluBench', args.datasets) + os.sep

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.file_name + '.jsonl')

    # ── Resume ───────────────────────────────────────────────────────────────
    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line.strip())['id'])
                except Exception:
                    pass
        print(f'[resume] skipping {len(done_ids)} already-done samples', flush=True)

    # ── Model ────────────────────────────────────────────────────────────────
    from transformers import AutoModel, AutoProcessor
    print(f'[model] loading {args.model_path} (bf16, sdpa)...', flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation='eager')  # the omni wrapper class does not declare sdpa support; the inner NemotronH LLM still runs sdpa
    DEVICE = os.environ.get('NEMO_DEVICE', 'cuda:0')
    model.to(DEVICE).eval()
    _repair_buffers(model, args.model_path, DEVICE)
    # EVS video pruning assumes a SQUARE token grid; our dynamic-resize grids
    # are non-square (e.g. 12x21=252), which crashes evs.py reshape. Pruning is
    # a throughput knob (drops redundant tokens); 0.0 keeps ALL video tokens =
    # strictly more information, the conservative no-approximation setting.
    if getattr(model, 'video_pruning_rate', 0):
        print(f"[repair] video_pruning_rate {model.video_pruning_rate} -> 0.0 "
              f"(EVS square-grid bug, keep all tokens)", flush=True)
        model.video_pruning_rate = 0.0
    print(f'[model] loaded in {time.time()-t0:.0f}s', flush=True)
    tokenizer = processor.tokenizer

    data = load_dataset(args.data_path, args.data_root)
    if args.dry_run:
        data = data[:3]
        print(f'[dry_run] processing {len(data)} samples', flush=True)

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)
    all_in = all_out = 0
    start = time.time()

    for idx, item in enumerate(data):
        if item['id'] in done_ids:
            continue
        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        # ── Media ────────────────────────────────────────────────────────────
        json_dir = os.path.dirname(args.data_path)
        videos = images = audio = None
        has_video = has_audio = has_image = False
        if item.get('video_path'):
            vp = resolve_media_path(item['video_path'], json_dir, args.data_root)
            videos = load_video_frames_native(vp)
            has_video = videos is not None
            if not has_video:
                print(f'  [MEDIA WARN] video unreadable: {vp}', flush=True)
        if item.get('audio_path'):
            ap = resolve_media_path(item['audio_path'], json_dir, args.data_root)
            audio = load_audio_wave(ap)
            has_audio = audio is not None
            if not has_audio:
                print(f'  [MEDIA WARN] audio unreadable: {ap}', flush=True)
        if item.get('image_path'):
            ip = resolve_media_path(item['image_path'], json_dir, args.data_root)
            images = load_image(ip)
            has_image = images is not None
            if not has_image:
                print(f'  [MEDIA WARN] image unreadable: {ip}', flush=True)

        # ── Prompt with literal placeholders ─────────────────────────────────
        user_prompt = get_user_prompt(
            item['question'], args.datasets, args.prompt_version,
            passage=item.get('passage'))
        media_prefix = ''
        if has_video:
            media_prefix += '<video>\n'
        if has_audio:
            # the processor replaces the LITERAL '<so_embedding>' (its audio_token),
            # NOT '<audio>' as the docstring claims
            media_prefix += '<so_embedding>\n'
        if has_image:
            media_prefix += '<image>\n'
        user_text = (media_prefix + user_prompt) if media_prefix else user_prompt
        messages = [{'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_text}]
        try:
            text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=args.enable_thinking)
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)

        proc_kwargs = dict(text=[text], return_tensors='pt')
        if has_image:
            proc_kwargs['images'] = images
        if has_video:
            proc_kwargs['videos'] = videos
        if has_audio:
            proc_kwargs['audio'] = audio

        gen_ids = None
        logp_list = None
        input_len = 0
        try:
            inputs = processor(**proc_kwargs)
            inputs = {k: (v.to(DEVICE) if torch.is_tensor(v) else v)
                      for k, v in inputs.items()}
            if inputs.get('pixel_values_videos') is not None:
                _pv = inputs['pixel_values_videos']
                print(f'  [video] pv={tuple(_pv.shape)} dtype={_pv.dtype} '
                      f'num_tokens={inputs.get("num_tokens")} num_patches={inputs.get("num_patches")}', flush=True)
            if inputs.get('pixel_values') is not None:
                _pi = inputs['pixel_values']
                print(f'  [image] pv={tuple(_pi.shape)} dtype={_pi.dtype}', flush=True)
            for _meta_key in ('num_patches', 'num_tokens', 'imgs_sizes'):
                inputs.pop(_meta_key, None)   # processor metadata, not model kwargs
            input_len = int(inputs['input_ids'].shape[1])
            all_in += input_len
            if has_audio:
                clips = inputs.get('sound_clips')
                n_s = int(np.asarray(clips[0]).shape[-1]) if clips is not None else 0
                print(f'  [audio] {n_s} samples @16kHz '
                      f'({"non-empty => audio really in" if n_s > 0 else "EMPTY!"})',
                      flush=True)

            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                pad_token_id=tokenizer.pad_token_id or 0,
            )
            if args.record_process:
                gen_kwargs.update(output_scores=True, return_dict_in_generate=True)
            with torch.inference_mode():
                out = model.generate(**inputs, **gen_kwargs)

            if args.record_process and hasattr(out, 'sequences'):
                gen_ids = out.sequences[0][input_len:]
                logp_list = token_logprobs(gen_ids.cpu(), out.scores)
            elif hasattr(out, 'sequences'):
                gen_ids = out.sequences[0][input_len:]
            else:
                gen_ids = out[0][input_len:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            all_out += len(gen_ids)
            elapsed = time.time() - t0
            print(f'  response[:120]: {response[:120]!r}', flush=True)
            print(f'  tokens_out={len(gen_ids)} | time={elapsed:.1f}s', flush=True)
        except Exception as e:
            import traceback
            print(f'  [ERROR] {e}', flush=True)
            traceback.print_exc()
            response = ''
            elapsed = time.time() - t0
            gen_ids = None

        # NEMO_KEEP_CACHE=1 keeps the caching-allocator reservation (see
        # warm_claim.py): the freed blocks are what concurrent tenants grab
        # when this farm is contended. Default (unset) keeps the proven path.
        if not os.environ.get('NEMO_KEEP_CACHE'):
            torch.cuda.empty_cache()

        result = make_result(item, response, args.datasets, 'base_model_nemotron',
                             time_sec=elapsed)
        result['nemotron'] = {
            'runtime': 'transformers-5.6.2-trust-remote-code',
            'enable_thinking': bool(args.enable_thinking),
            'media': {'video': has_video, 'audio': has_audio, 'image': has_image},
        }
        if args.record_process:
            result['record_process'] = True
            if gen_ids is not None and logp_list is not None:
                cum = float(np.sum(logp_list))
                result['token_ids'] = gen_ids.tolist()
                result['logp_list'] = logp_list
                result['cumulative_logprob'] = round(cum, 4)
                result['norm_logp'] = round(cum / (len(gen_ids) + 1e-8), 4)
                result['input_len'] = input_len
                result['output_tokens'] = int(len(gen_ids))
            else:
                result['token_ids'] = None
                result['logp_list'] = None
                result['cumulative_logprob'] = None
                result['norm_logp'] = None
                result['input_len'] = input_len
                result['output_tokens'] = 0
        save_result(result, output_path)

    total_time = time.time() - start
    with open(os.path.join(args.time_path, args.file_name + '.txt'), 'w') as f:
        f.write(f'total_time: {total_time:.2f}s\n')
        f.write(f'total_input_tokens: {all_in}\n')
        f.write(f'total_output_tokens: {all_out}\n')
    print(f'\n[done] total={total_time:.1f}s | output={output_path}', flush=True)


if __name__ == '__main__':
    main()
