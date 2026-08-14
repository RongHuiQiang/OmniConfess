# -*- coding: utf-8 -*-
"""
base_cad_minicpm.py -- MiniCPM-o-2.6 CAD (Context-Aware Decoding) baseline.

What CAD is
-----------
Shi et al. 2024, "Trusting Your Evidence: Hallucinate Less with Context-aware
Decoding". Core formula (Eq. 4 of the paper):

    y_t ~ softmax[ (1+alpha) * logit(y_t | c, x, y_<t) - alpha * logit(y_t | x, y_<t) ]

That is, the logits of a branch with the evidence c and a branch without it are
contrasted, amplifying the increment the evidence contributes and suppressing what
the model could produce from its parametric prior alone.

Relation to VCD
---------------
The formula is identical; only the construction of the perturbed branch differs:
  VCD keeps the evidence but corrupts it with Gaussian noise on pixel_values;
  CAD removes the evidence entirely.
Both therefore share shared/minicpm_decode.make_contrast_step_fn.

Removing the evidence is handled per modality:
  image datasets (haloquest / phd / cmm)  the auxiliary branch carries no image;
  audio-visual datasets (avhbench)        the auxiliary branch carries no
                                          audio-visual block;
  text-only datasets (ragtruth / pubmedqa) the auxiliary branch's user prompt has
                                          passage=None, leaving the text section
                                          empty.
So CAD does not degenerate to greedy on the text-only sets. That is precisely the
key difference from VCD, and the reason it is listed as a separate baseline.

Why model.chat() cannot be used
-------------------------------
As with base_vcd_minicpm.py: chat() runs beam search internally and discards
logits_processor, so per-token logits are unavailable. This script therefore uses
shared/minicpm_decode to reproduce chat()'s preprocessing and then builds its own
step-by-step greedy loop.

On the contrast domain
----------------------
The CAD paper works on raw logits; this script defaults to log_domain=True, taking
log_softmax of each branch first. Under greedy argmax the two are exactly
equivalent, since log_softmax only subtracts a scalar logsumexp from each branch
and the weighted combination is still a constant offset applied uniformly across
the vocabulary, which cannot change the argmax. The log domain is chosen to match
the Qwen-side base_vcd.py formulation for easier cross-checking. The two only
diverge when apc_beta > 0 or when sampling is used.
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
    get_terminators, greedy_generate, make_contrast_step_fn,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_PATH = "openbmb/MiniCPM-o-2_6"

DEFAULT_ALPHA = 0.5


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


def build_content_for_item(item, data_path, data_root, dataset_name, prompt_version,
                           drop_evidence=False):
    """
    With drop_evidence=True, build the CAD auxiliary branch input: no media at all
    and an empty passage. Everything else (question text, prompt template, system
    prompt) matches the main branch, so the only difference between the two
    branches is whether the evidence is present.
    """
    json_dir = os.path.dirname(data_path)

    # --- Omni mode: video + audio (avhbench) ---
    if not drop_evidence and item.get('video_path') and item.get('audio_path'):
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
    if not drop_evidence and item.get('image_path'):
        ip = resolve_media_path(item['image_path'], json_dir, data_root)
        if os.path.exists(ip):
            content.append(Image.open(ip).convert('RGB'))

    user_prompt = get_user_prompt(
        item['question'], dataset_name, prompt_version,
        passage=None if drop_evidence else item.get('passage'))
    content.append(user_prompt)

    return content, False


def has_evidence(item):
    """Whether the main branch actually contains evidence that could be removed. Without it CAD has nothing to contrast."""
    return bool(item.get('image_path') or item.get('video_path')
                or item.get('audio_path') or item.get('passage'))


def _text_only_msg(msg):
    """
    Strip the media content from a message, keeping only the text.

    The CAD auxiliary branch does not go through the omni path, so a PIL.Image or
    np.ndarray smuggled in via the system prompt (the omni system prompt may carry
    reference audio) would make build_minicpm_inputs wrongly infer use_tts_template.
    """
    content = msg.get('content')
    if isinstance(content, str):
        return dict(msg)
    if isinstance(content, (list, tuple)):
        return {**msg, 'content': [c for c in content if isinstance(c, str)]}
    return dict(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./OmniHalluBench/')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--file_name', type=str, default='minicpm_cad_out')
    parser.add_argument('--time_path', type=str, default='./results/time')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--max_samples', type=int, default=0,
                        help='0 = all samples')
    parser.add_argument('--prompt_version', type=str, default='v0')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                        help='CAD contrast weight (default 0.5, the value Shi et al. 2024 '
                             'use for summarization; the paper uses 1.0 for knowledge conflict)')
    parser.add_argument('--apc_beta', type=float, default=0.0,
                        help='Adaptive plausibility constraint beta (0 = off; CAD as '
                             'published applies no APC)')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='1.0 = off. Note model.chat(sampling=False) uses 1.2 with '
                             'num_beams=3; greedy+1.0 matches the Qwen-side baselines.')
    args = parser.parse_args()

    print(f"[cad] alpha={args.alpha}, apc_beta={args.apc_beta}, "
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

    step_fn = make_contrast_step_fn(alpha=args.alpha, apc_beta=args.apc_beta,
                                    log_domain=True)
    terminators = get_terminators(model, tokenizer)

    data = load_dataset(args.data_path, args.data_root)
    if args.max_samples > 0:
        data = data[:args.max_samples]

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)
    start = time.time()
    n_with_cad = 0

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

            # -- CAD auxiliary branch: same question and template, evidence removed --
            aux_branches = None
            if has_evidence(item):
                try:
                    aux_content, _ = build_content_for_item(
                        item, args.data_path, args.data_root,
                        args.datasets, args.prompt_version, drop_evidence=True)
                    aux_msgs = [{'role': 'user', 'content': aux_content}]
                    if len(msgs) > 1:
                        aux_msgs.insert(0, _text_only_msg(msgs[0]))
                    aux_inputs = build_minicpm_inputs(
                        model, model.processor, aux_msgs,
                        omni_input=False, max_slice_nums=1, use_image_id=False)
                    aux_branches = [{
                        'embeds': get_multimodal_embeds(model, aux_inputs),
                        'attention_mask': aux_inputs.get('attention_mask'),
                    }]
                    n_with_cad += 1
                except Exception as ae:
                    print(f'  [WARN] CAD aux branch failed ({ae}); falling back to greedy',
                          flush=True)
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
            print(f'  cad={"on" if aux_branches else "off (no evidence to drop)"}', flush=True)
            print(f'  response[:120]: {response[:120]!r}', flush=True)
            print(f'  time={elapsed:.1f}s', flush=True)

        except Exception as e:
            print(f'  [ERROR] {e}', flush=True)
            response = ''
            elapsed = time.time() - t0

        torch.cuda.empty_cache()

        result = make_result(item, response, args.datasets, 'minicpm_cad',
                             time_sec=elapsed)
        save_result(result, output_path)

    total_time = time.time() - start
    n_done = len(data) - len(done_ids)
    print(f'\n[done] {n_done} samples in {total_time/60:.1f} min '
          f'({n_with_cad} with an active CAD branch). Results: {output_path}', flush=True)

    summary = os.path.join(args.time_path, args.file_name + '.txt')
    with open(summary, 'w') as f:
        f.write(f'total_time: {total_time:.2f}s\n')
        f.write(f'samples: {n_done}\n')
        f.write(f'samples_with_cad_branch: {n_with_cad}\n')
        f.write(f'alpha: {args.alpha}\n')
        f.write(f'apc_beta: {args.apc_beta}\n')
        f.write(f'repetition_penalty: {args.repetition_penalty}\n')
        f.write('decode: greedy, per-token CAD contrast (context-free aux branch, '
                'log-prob domain, bypasses model.chat)\n')


if __name__ == '__main__':
    main()
