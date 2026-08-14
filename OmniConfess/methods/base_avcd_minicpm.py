# -*- coding: utf-8 -*-
"""
base_avcd_minicpm.py -- MiniCPM-o-2.6 AVCD (Audio-Visual Contrastive Decoding) baseline.

What AVCD is
------------
Contrastive decoding for audio-visual large models. It first determines which
modality the model is mainly relying on for the current sample (language, video or
audio), then builds three auxiliary branches that weaken the non-dominant
modalities, and uses a four-term combination to amplify the evidence outside the
dominant modality, suppressing hallucinations caused by one modality running ahead
of the others.

Core formula:
    (2+2a)*orig - 2a*o1 + o2 + o3
where o1 masks both non-dominant modalities and o2/o3 each mask one of them. An
adaptive plausibility constraint is then applied, and an entropy gate decides
whether the contrast runs at all (below an entropy of 0.6 it decodes greedily).

Hyperparameters
---------------
  alpha = 2.5   the value the official code uses for the AVHBench val/test splits.
                Note that the official function signature defaults to 0.5, but the
                AVHBench experiment script passes 2.5 explicitly; AVHBench is the
                main setting here, so 2.5 is used.
  beta  = 0.2   hardcoded in the official implementation
  entropy threshold 0.6   hardcoded in the official implementation
  APC fill value -1e-4    hardcoded in the official implementation. This is a
                suspicious choice, since a normal APC fills with -inf, and when the
                maximum logit of the original branch is negative it lets a rejected
                candidate overtake. It is reproduced faithfully here, but the
                number of such apc_escape events is recorded so it can be checked;
                pass --apc_fill -inf for the corrected variant.

Porting decisions
-----------------
The official code hardcodes the token layout as VideoLLaMA2's fixed spans, whereas
MiniCPM-o's visual and audio token counts vary with the input, so the spans are
derived dynamically from the processor's image_bound and audio_bounds.
Masking is changed from "zero the attention weights and renormalise" to "zero the
attention_mask", which softmax renormalises naturally; the two are equivalent.
Steps skipped by the entropy gate accumulate their tokens and feed them in one go
on the next real step, which is output-equivalent but avoids the official code's
O(n^2) cost of recomputing the whole sequence at every step.

Non-audio-visual datasets
-------------------------
AVCD's three-branch design assumes language, video and audio are all present.
Image datasets have no audio, and text-only datasets have neither:
  - image datasets: the audio index is empty, so the "mask audio" branch
    degenerates to the main branch. The formula still holds but loses one effective
    contrast term, and the log records the modalities actually found.
  - text only: there are no media tokens at all, so AVCD cannot build any branch
    and degenerates to greedy automatically.
This is an inherent limitation of applying an audio-visual method to unimodal data,
recorded rather than hidden.

Supports: avhbench (audio+video, AVCD's intended setting), haloquest / phd / cmm
(image, degenerates to two modalities), ragtruth / pubmedqa (text, degenerates to
greedy).
"""
import os, sys, time, argparse, json, random, math
import torch
import numpy as np
import librosa
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from shared.data_loader import load_dataset, resolve_media_path
from shared.eval_utils import make_result, save_result
from shared.prompts import get_system_prompt, get_user_prompt
from shared.minicpm_decode import (
    patch_smart_apply, build_minicpm_inputs, get_multimodal_embeds,
    get_terminators, get_modality_indices, avcd_generate,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_PATH = "openbmb/MiniCPM-o-2_6"

DEFAULT_ALPHA = 2.5
DEFAULT_BETA = 0.2
DEFAULT_ENTROPY = 0.6
DEFAULT_APC_FILL = -1e-4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _patch_smart_apply():
    """Fix transformers 4.52+ incompatibility with MiniCPM-o custom modules."""
    patch_smart_apply()


def load_minicpm_model():
    from transformers import AutoModel, AutoTokenizer
    _patch_smart_apply()
    print(f'[model] loading MiniCPM-o-2.6 from {MODEL_PATH}...', flush=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=True,
        init_tts=False,
    )
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f'[model] loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB', flush=True)
    return model, tokenizer


def build_omni_content(video_path, audio_path):
    """Build per-second <unit> chunks for MiniCPM-o omni mode (video+audio)."""
    import cv2
    audio_np, sr = librosa.load(audio_path, sr=16000, mono=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    num_units = math.ceil(duration)

    contents = []
    for i in range(num_units):
        frame_idx = min(int((i + 1) * fps), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            image = Image.new('RGB', (224, 224))
        audio_chunk = audio_np[sr * i:sr * (i + 1)]
        if len(audio_chunk) == 0:
            audio_chunk = np.zeros(sr, dtype=np.float32)
        contents.extend(["<unit>", image, audio_chunk])
    cap.release()
    print(f'  [omni] {num_units} units from {duration:.1f}s video', flush=True)
    return contents


def build_content_for_item(item, data_path, data_root, dataset_name, prompt_version):
    """Build MiniCPM-o message content list. Returns (content_list, is_omni)."""
    json_dir = os.path.dirname(data_path)

    if item.get('video_path') and item.get('audio_path'):
        vp = resolve_media_path(item['video_path'], json_dir, data_root)
        ap = resolve_media_path(item['audio_path'], json_dir, data_root)
        if os.path.exists(vp) and os.path.exists(ap):
            content = build_omni_content(vp, ap)
            user_prompt = get_user_prompt(
                item['question'], dataset_name, prompt_version,
                passage=item.get('passage'))
            content.append(user_prompt)
            return content, True

    content = []
    if item.get('image_path'):
        ip = resolve_media_path(item['image_path'], json_dir, data_root)
        if os.path.exists(ip):
            content.append(Image.open(ip).convert('RGB'))

    user_prompt = get_user_prompt(
        item['question'], dataset_name, prompt_version,
        passage=item.get('passage'))
    content.append(user_prompt)

    return content, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./OmniHalluBench/')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--file_name', type=str, default='minicpm_avcd_out')
    parser.add_argument('--time_path', type=str, default='./results/time')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--max_samples', type=int, default=0,
                        help='0 = all samples')
    parser.add_argument('--prompt_version', type=str, default='v0')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                        help='AVCD contrast weight (default 2.5, the value the official '
                             'AVHBench scripts pass: inference_AVH_val.py:112)')
    parser.add_argument('--apc_beta', type=float, default=DEFAULT_BETA,
                        help='Adaptive plausibility constraint beta '
                             '(default 0.2, hard-coded at AVCD __init__.py:164)')
    parser.add_argument('--apc_fill', type=float, default=DEFAULT_APC_FILL,
                        help='Value written into logits of implausible candidates. '
                             'Default -1e-4 reproduces AVCD __init__.py:231 verbatim; '
                             'pass -inf for the textbook APC behaviour.')
    parser.add_argument('--entropy_threshold', type=float, default=DEFAULT_ENTROPY,
                        help='Below this entropy the step stays plain greedy '
                             '(default 0.6, hard-coded at AVCD __init__.py:171)')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='1.0 = off. Note model.chat(sampling=False) uses 1.2 with '
                             'num_beams=3; greedy+1.0 matches the Qwen-side baselines.')
    args = parser.parse_args()

    print(f"[avcd] alpha={args.alpha}, apc_beta={args.apc_beta}, "
          f"apc_fill={args.apc_fill}, entropy_threshold={args.entropy_threshold}, "
          f"repetition_penalty={args.repetition_penalty}", flush=True)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)

    output_path = os.path.join(args.output_dir, args.file_name + '.jsonl')

    # Resume
    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line.strip())['id'])
                except Exception:
                    pass
        if done_ids:
            print(f'[resume] skipping {len(done_ids)} already-done samples', flush=True)

    model, tokenizer = load_minicpm_model()
    terminators = get_terminators(model, tokenizer)

    data = load_dataset(args.data_path, args.data_root)
    if args.max_samples > 0:
        data = data[:args.max_samples]

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)
    start = time.time()
    n_active, n_gated, n_steps, n_escape = 0, 0, 0, 0
    dom_counter = {}

    for idx, item in enumerate(data):
        if item['id'] in done_ids:
            continue

        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        try:
            content, is_omni = build_content_for_item(
                item, args.data_path, args.data_root,
                args.datasets, args.prompt_version)

            msgs = [{'role': 'user', 'content': content}]
            if is_omni:
                msgs.insert(0, model.get_sys_prompt(mode='omni', language='en'))
            elif system_prompt:
                msgs.insert(0, {'role': 'system', 'content': [system_prompt]})

            # Same preprocessing as chat(); only the decoding differs.
            inputs = build_minicpm_inputs(
                model, model.processor, msgs,
                omni_input=is_omni, max_slice_nums=1, use_image_id=False)
            embeds = get_multimodal_embeds(model, inputs)
            attn_mask = inputs.get('attention_mask')

            v_idx, a_idx = get_modality_indices(
                inputs.get('input_ids'), embeds.device,
                image_bound=inputs.get('image_bound'),
                audio_bounds=inputs.get('audio_bounds'))

            mods = ('V' if v_idx is not None else '') + ('A' if a_idx is not None else '')
            print(f'  modalities={mods or "text-only"} '
                  f'(v={0 if v_idx is None else v_idx.numel()} '
                  f'a={0 if a_idx is None else a_idx.numel()} tokens)', flush=True)

            response, _, stats = avcd_generate(
                model, tokenizer, embeds,
                attention_mask=attn_mask,
                v_idx=v_idx, a_idx=a_idx,
                alpha=args.alpha, apc_beta=args.apc_beta,
                apc_fill=args.apc_fill,
                entropy_threshold=args.entropy_threshold,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                terminators=terminators,
            )

            if stats['avcd_active']:
                n_active += 1
                dom_counter[stats['dominant']] = dom_counter.get(stats['dominant'], 0) + 1
            n_gated += stats['gated_steps']
            n_steps += stats['total_steps']
            n_escape += stats['apc_escape']

            elapsed = time.time() - t0
            print(f'  avcd={"on" if stats["avcd_active"] else "off (no media tokens)"}'
                  f' dominant={stats["dominant"]}'
                  f' gated={stats["gated_steps"]}/{stats["total_steps"]}'
                  f' apc_escape={stats["apc_escape"]}', flush=True)
            print(f'  response[:120]: {response[:120]!r}', flush=True)
            print(f'  time={elapsed:.1f}s', flush=True)

        except Exception as e:
            print(f'  [ERROR] {e}', flush=True)
            response = ''
            elapsed = time.time() - t0

        torch.cuda.empty_cache()

        result = make_result(item, response, args.datasets, 'minicpm_avcd',
                             time_sec=elapsed)
        save_result(result, output_path)

    total_time = time.time() - start
    n_done = len(data) - len(done_ids)
    print(f'\n[done] {n_done} samples in {total_time/60:.1f} min '
          f'({n_active} with AVCD active, {n_gated}/{n_steps} steps gated). '
          f'Results: {output_path}', flush=True)

    summary = os.path.join(args.time_path, args.file_name + '.txt')
    with open(summary, 'w') as f:
        f.write(f'total_time: {total_time:.2f}s\n')
        f.write(f'samples: {n_done}\n')
        f.write(f'samples_with_avcd_active: {n_active}\n')
        f.write(f'gated_steps: {n_gated}\n')
        f.write(f'total_steps: {n_steps}\n')
        f.write(f'apc_escape_steps: {n_escape}\n')
        f.write(f'dominant_modality_counts: {json.dumps(dom_counter)}\n')
        f.write(f'alpha: {args.alpha}\n')
        f.write(f'apc_beta: {args.apc_beta}\n')
        f.write(f'apc_fill: {args.apc_fill}\n')
        f.write(f'entropy_threshold: {args.entropy_threshold}\n')
        f.write(f'repetition_penalty: {args.repetition_penalty}\n')
        f.write('decode: greedy, entropy-gated AVCD 4-term contrast over 3 '
                'modality-masked branches (bypasses model.chat)\n')


if __name__ == '__main__':
    main()
