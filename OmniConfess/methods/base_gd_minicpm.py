# -*- coding: utf-8 -*-
"""
base_gd_minicpm.py — MiniCPM-o-2.6 Guided Decoding baseline (simplified).

Algorithm (adapted for chat-only API):
  Phase 1: Generate num_rollout candidate responses via sampling (temperature=0.6)
  Phase 2: Judge each candidate with a correctness prompt -> score 0.95 or 0.05
  Phase 3: Select candidate with highest judge score (tie-break: first occurrence)

Usage:
    CUDA_VISIBLE_DEVICES=1 python base_gd_minicpm.py \
        --datasets avhbench --data_path ./OmniHalluBench/avhbench/avh_converted.json \
        --output_dir ./results --file_name minicpm_gd_avh_v0 --max_samples 2
"""
import os, sys, time, argparse, json, random, math
import torch
import numpy as np
import librosa
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.data_loader import load_dataset, resolve_media_path
from shared.eval_utils import make_result, save_result
from shared.prompts import get_system_prompt, get_user_prompt

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_PATH = "./models/MiniCPM-o-2.6"

GD_JUDGE_PROMPT = (
    "Is the following answer logically correct and consistent with the "
    "evidence in the media? Answer ONLY with (A) Correct or (B) Incorrect.\n\n"
    "Answer to evaluate: {answer}"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _patch_smart_apply():
    """Fix transformers 4.52+ incompatibility with MiniCPM-o custom modules."""
    import torch
    def patched_smart_apply(self, fn):
        for module in self.children():
            if hasattr(module, "_init_weights"):
                if hasattr(module, "_initialize_weights"):
                    module.smart_apply(module._initialize_weights)
                else:
                    module.smart_apply(fn)
            else:
                module.smart_apply(fn)
        fn(self)
        return self
    torch.nn.Module.smart_apply = patched_smart_apply


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
    # Load audio
    audio_np, sr = librosa.load(audio_path, sr=16000, mono=True)
    # Get video duration and fps
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    num_units = math.ceil(duration)

    contents = []
    for i in range(num_units):
        # Extract 1 frame at second (i+1) or last frame
        frame_idx = min(int((i + 1) * fps), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            image = Image.new('RGB', (224, 224))
        # Extract 1s audio chunk
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
            # Append text prompt at the end
            user_prompt = get_user_prompt(
                item['question'], dataset_name, prompt_version,
                passage=item.get('passage'))
            content.append(user_prompt)
            return content, True

    # --- Standard mode: image / text ---
    content = []

    # Image (haloquest, phd, cmm)
    if item.get('image_path'):
        ip = resolve_media_path(item['image_path'], json_dir, data_root)
        if os.path.exists(ip):
            content.append(Image.open(ip).convert('RGB'))

    # Text prompt
    user_prompt = get_user_prompt(
        item['question'], dataset_name, prompt_version,
        passage=item.get('passage'))
    content.append(user_prompt)

    return content, False


def build_judge_content_for_item(item, data_path, data_root, dataset_name, candidate_answer):
    """Build judge message content: same media + judge prompt instead of question."""
    json_dir = os.path.dirname(data_path)
    judge_text = GD_JUDGE_PROMPT.format(answer=candidate_answer)

    # --- Omni mode: video + audio (avhbench) ---
    if item.get('video_path') and item.get('audio_path'):
        vp = resolve_media_path(item['video_path'], json_dir, data_root)
        ap = resolve_media_path(item['audio_path'], json_dir, data_root)
        if os.path.exists(vp) and os.path.exists(ap):
            content = build_omni_content(vp, ap)
            content.append(judge_text)
            return content, True

    # --- Standard mode: image / text ---
    content = []

    if item.get('image_path'):
        ip = resolve_media_path(item['image_path'], json_dir, data_root)
        if os.path.exists(ip):
            content.append(Image.open(ip).convert('RGB'))

    content.append(judge_text)
    return content, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./OmniHalluBench/')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--file_name', type=str, default='minicpm_gd_out')
    parser.add_argument('--time_path', type=str, default='./results/time')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--max_samples', type=int, default=0,
                        help='0 = all samples')
    parser.add_argument('--num_rollout', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--prompt_version', type=str, default='v0')
    args = parser.parse_args()

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

    data = load_dataset(args.data_path, args.data_root)
    if args.max_samples > 0:
        data = data[:args.max_samples]

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)
    start = time.time()

    for idx, item in enumerate(data):
        if item['id'] in done_ids:
            continue

        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        try:
            content, is_omni = build_content_for_item(
                item, args.data_path, args.data_root,
                args.datasets, args.prompt_version)

            # ── Phase 1: Generate num_rollout candidates with sampling ───────
            candidates = []
            for r in range(args.num_rollout):
                msgs = [{'role': 'user', 'content': content}]

                if is_omni:
                    omni_sys = model.get_sys_prompt(mode='omni', language='en')
                    msgs.insert(0, omni_sys)
                    response = model.chat(
                        msgs=msgs,
                        tokenizer=tokenizer,
                        sampling=True,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        omni_input=True,
                        use_tts_template=False,
                        generate_audio=False,
                        max_slice_nums=1,
                        use_image_id=False,
                        return_dict=False,
                    )
                else:
                    if system_prompt:
                        msgs.insert(0, {'role': 'system', 'content': [system_prompt]})
                    response = model.chat(
                        msgs=msgs,
                        tokenizer=tokenizer,
                        sampling=True,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                    )

                candidates.append(response)
                torch.cuda.empty_cache()

            print(f'  [phase1] generated {len(candidates)} candidates', flush=True)

            # ── Phase 2: Judge each candidate ────────────────────────────────
            judge_scores = []
            judge_verdicts = []
            for c_idx, candidate in enumerate(candidates):
                judge_content, judge_is_omni = build_judge_content_for_item(
                    item, args.data_path, args.data_root,
                    args.datasets, candidate)

                judge_msgs = [{'role': 'user', 'content': judge_content}]

                if judge_is_omni:
                    omni_sys = model.get_sys_prompt(mode='omni', language='en')
                    judge_msgs.insert(0, omni_sys)
                    verdict = model.chat(
                        msgs=judge_msgs,
                        tokenizer=tokenizer,
                        sampling=False,
                        max_new_tokens=20,
                        omni_input=True,
                        use_tts_template=False,
                        generate_audio=False,
                        max_slice_nums=1,
                        use_image_id=False,
                        return_dict=False,
                    )
                else:
                    if system_prompt:
                        judge_msgs.insert(0, {'role': 'system', 'content': [system_prompt]})
                    verdict = model.chat(
                        msgs=judge_msgs,
                        tokenizer=tokenizer,
                        sampling=False,
                        max_new_tokens=20,
                    )

                score = 0.95 if '(A)' in verdict or 'Correct' in verdict else 0.05
                judge_scores.append(score)
                judge_verdicts.append(verdict)
                torch.cuda.empty_cache()

            print(f'  [phase2] judge scores: {judge_scores}', flush=True)

            # ── Phase 3: Select best candidate (tie-break: first) ────────────
            best_idx = int(np.argmax(judge_scores))
            final_response = candidates[best_idx]

            # Store all candidates and judge results in cot
            cot_entries = []
            for c_idx in range(len(candidates)):
                cot_entries.append(
                    f"[Candidate {c_idx+1}] (score={judge_scores[c_idx]:.2f})\n"
                    f"{candidates[c_idx]}\n"
                    f"[Judge verdict]: {judge_verdicts[c_idx]}"
                )
            cot = "\n\n---\n\n".join(cot_entries)
            cot += f"\n\n=== Selected: Candidate {best_idx+1} ==="

            elapsed = time.time() - t0
            print(f'  response[:120]: {final_response[:120]!r}', flush=True)
            print(f'  best_idx={best_idx} | time={elapsed:.1f}s', flush=True)

        except Exception as e:
            print(f'  [ERROR] {e}', flush=True)
            final_response = ''
            cot = ''
            elapsed = time.time() - t0

        torch.cuda.empty_cache()

        result = make_result(item, final_response, args.datasets, 'minicpm_gd',
                             time_sec=elapsed, cot=cot)
        save_result(result, output_path)

    total_time = time.time() - start
    n_done = len(data) - len(done_ids)
    print(f'\n[done] {n_done} samples in {total_time/60:.1f} min. Results: {output_path}', flush=True)

    summary = os.path.join(args.time_path, args.file_name + '.txt')
    with open(summary, 'w') as f:
        f.write(f'total_time: {total_time:.2f}s\n')
        f.write(f'samples: {n_done}\n')
        f.write(f'num_rollout: {args.num_rollout}\n')
        f.write(f'temperature: {args.temperature}\n')


if __name__ == '__main__':
    main()
