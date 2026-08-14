# -*- coding: utf-8 -*-
"""
base_vcd_minicpm.py -- MiniCPM-o-2.6 VCD (Visual Contrastive Decoding) baseline.

Implementation
--------------
VCD needs the logits of a clean-visual branch and a noised-visual branch at every
token. MiniCPM-o's model.chat() cannot do that: it runs beam search internally
(num_beams=3, repetition_penalty=1.2) and only forwards keys already present in the
generation config, so logits_processor is silently dropped. This script therefore
bypasses chat():

1. shared/minicpm_decode.build_minicpm_inputs reproduces chat()'s preprocessing
   line for line, guaranteeing the inputs match base_model_minicpm.py exactly;
2. noise_visual_inputs adds Gaussian noise to pixel_values to build the perturbed
   branch;
3. the two branches prefill separately and keep their own KV caches, and each step
   feeds both the same already-selected token;
4. each step applies the VCD contrast.

The hyperparameters and the formula follow the Qwen-side base_vcd.py: alpha = 0.1,
Gaussian noise with std = 0.1 clamped to [-10, 10], contrast in the log-probability
domain, and no adaptive plausibility constraint.

The text-only datasets (ragtruth, pubmedqa) have no pixel_values, so the perturbed
branch is empty and decoding degenerates to greedy, matching the TEXT_ONLY_DS
behaviour on the Qwen side.

Note that this script was historically a stub that degenerated to greedy: --alpha
and --noise_std were declared but marked unused, and its output was byte-identical
to base_model.
"""
import os, sys, time, argparse, json, random, math
import torch
import numpy as np
import librosa
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from shared.paths import resolve_model
from shared.data_loader import load_dataset, resolve_media_path
from shared.eval_utils import make_result, save_result
from shared.prompts import get_system_prompt, get_user_prompt
from shared.minicpm_decode import (
    patch_smart_apply, build_minicpm_inputs, get_multimodal_embeds,
    get_terminators, greedy_generate, noise_visual_inputs, make_contrast_step_fn,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_PATH = resolve_model("openbmb/MiniCPM-o-2_6")

DEFAULT_ALPHA = 0.1
DEFAULT_NOISE_STD = 0.1


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
    """Build MiniCPM-o message content list.

    For video+audio (avhbench): uses omni mode with per-second <unit> chunks.
    For image-only / text-only: uses standard content list.
    Returns (content_list, is_omni).
    """
    json_dir = os.path.dirname(data_path)

    # --- Omni mode: video + audio (avhbench) ---
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

    # --- Standard mode: image / text ---
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
    parser.add_argument('--file_name', type=str, default='minicpm_vcd_out')
    parser.add_argument('--time_path', type=str, default='./results/time')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--max_samples', type=int, default=0,
                        help='0 = all samples')
    parser.add_argument('--prompt_version', type=str, default='v0')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                        help='VCD contrast weight (default 0.1, matches base_vcd.py:37)')
    parser.add_argument('--noise_std', type=float, default=DEFAULT_NOISE_STD,
                        help='Gaussian noise std for visual distortion '
                             '(default 0.1, matches base_vcd.py:38)')
    parser.add_argument('--noise_mode', type=str, default='gaussian',
                        choices=['gaussian', 'diffusion', 'none'],
                        help="'gaussian' matches the Qwen-side baseline; 'diffusion' "
                             "matches the original VCD paper; 'none' builds no contrast "
                             "branch at all, degrading to plain greedy — that is the "
                             "matched-decoder control for base_model, which uses "
                             "model.chat()'s beam-3 + repetition_penalty 1.2, not greedy")
    parser.add_argument('--noise_step', type=int, default=500,
                        help='diffusion timestep, only used when --noise_mode diffusion')
    parser.add_argument('--apc_beta', type=float, default=0.0,
                        help='Adaptive plausibility constraint beta (0 = off, matches '
                             'base_vcd.py which applies no APC)')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='1.0 = off. Note model.chat(sampling=False) uses 1.2 with '
                             'num_beams=3; greedy+1.0 matches the Qwen-side baselines.')
    args = parser.parse_args()

    print(f"[vcd] alpha={args.alpha}, noise={args.noise_mode}(std={args.noise_std}), "
          f"apc_beta={args.apc_beta}, repetition_penalty={args.repetition_penalty}",
          flush=True)

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

    step_fn = make_contrast_step_fn(alpha=args.alpha, apc_beta=args.apc_beta,
                                    log_domain=True)
    terminators = get_terminators(model, tokenizer)

    data = load_dataset(args.data_path, args.data_root)
    if args.max_samples > 0:
        data = data[:args.max_samples]

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)
    start = time.time()
    n_with_vcd = 0

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

            noised = noise_visual_inputs(
                inputs, noise_mode=args.noise_mode,
                noise_std=args.noise_std, noise_step=args.noise_step)
            if noised is not None:
                aux_branches = [{
                    'embeds': get_multimodal_embeds(model, noised),
                    'attention_mask': attn_mask,
                }]
                n_with_vcd += 1
            else:
                aux_branches = None

            response, _ = greedy_generate(
                model, tokenizer, embeds,
                attention_mask=attn_mask,
                max_new_tokens=args.max_new_tokens,
                step_fn=step_fn,
                aux_branches=aux_branches,
                repetition_penalty=args.repetition_penalty,
                terminators=terminators,
            )

            elapsed = time.time() - t0
            print(f'  vcd={"on" if aux_branches else "off (text-only)"}', flush=True)
            print(f'  response[:120]: {response[:120]!r}', flush=True)
            print(f'  time={elapsed:.1f}s', flush=True)

        except Exception as e:
            print(f'  [ERROR] {e}', flush=True)
            response = ''
            elapsed = time.time() - t0

        torch.cuda.empty_cache()

        result = make_result(item, response, args.datasets, 'minicpm_vcd',
                             time_sec=elapsed)
        save_result(result, output_path)

    total_time = time.time() - start
    n_done = len(data) - len(done_ids)
    print(f'\n[done] {n_done} samples in {total_time/60:.1f} min '
          f'({n_with_vcd} with an active VCD branch). Results: {output_path}', flush=True)

    summary = os.path.join(args.time_path, args.file_name + '.txt')
    with open(summary, 'w') as f:
        f.write(f'total_time: {total_time:.2f}s\n')
        f.write(f'samples: {n_done}\n')
        f.write(f'samples_with_vcd_branch: {n_with_vcd}\n')
        f.write(f'alpha: {args.alpha}\n')
        f.write(f'noise_mode: {args.noise_mode}\n')
        f.write(f'noise_std: {args.noise_std}\n')
        f.write(f'apc_beta: {args.apc_beta}\n')
        f.write(f'repetition_penalty: {args.repetition_penalty}\n')
        f.write('decode: greedy, per-token VCD contrast in log-prob domain '
                '(bypasses model.chat)\n')


if __name__ == '__main__':
    main()
