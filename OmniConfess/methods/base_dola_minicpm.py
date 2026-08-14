# -*- coding: utf-8 -*-
"""
base_dola_minicpm.py -- MiniCPM-o-2.6 DoLa (Decoding by Contrasting Layers) baseline.

Implementation
--------------
DoLa needs per-token layer-wise logits, contrasting a mature layer against a
premature one. MiniCPM-o's model.chat() cannot provide them: it runs beam search
internally and discards logits_processor. This script therefore bypasses chat():

1. shared/minicpm_decode.build_minicpm_inputs reproduces chat()'s preprocessing
   line for line, guaranteeing the inputs match base_model_minicpm.py exactly;
2. a forward hook on the premature layer projects that layer's hidden state through
   model.llm.model.norm and model.llm.lm_head to obtain its logits;
3. generation proceeds greedily step by step, applying the DoLa contrast at each
   step.

The formula and the premature layer depth match the Qwen-side base_dola.py: both
language models have 28 layers, and premature = L14 is about 50% depth.

Note that this script was historically a stub that degenerated to greedy: --alpha
and --premature_layer were declared but never used, and its output was
byte-identical to base_model.
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
    get_terminators, PrematureLayerCapture, greedy_generate, make_dola_step_fn,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_PATH = "openbmb/MiniCPM-o-2_6"

PREMATURE_LAYER = 14
DEFAULT_ALPHA = 0.1


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./OmniHalluBench/')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--file_name', type=str, default='minicpm_dola_out')
    parser.add_argument('--time_path', type=str, default='./results/time')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--max_samples', type=int, default=0,
                        help='0 = all samples')
    parser.add_argument('--prompt_version', type=str, default='v0')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                        help='DoLa contrast weight (default 0.1, matches base_dola.py)')
    parser.add_argument('--premature_layer', type=int, default=PREMATURE_LAYER,
                        help='Premature layer index (default 14 ~ 50%% of 28 layers)')
    parser.add_argument('--apc_beta', type=float, default=0.0,
                        help='Adaptive plausibility constraint beta (0 = off)')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='1.0 = off. Note model.chat(sampling=False) uses 1.2 with '
                             'num_beams=3; greedy+1.0 matches the Qwen-side baselines.')
    args = parser.parse_args()

    print(f"[dola] premature_layer=L{args.premature_layer}, alpha={args.alpha}, "
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

    n_layers = len(model.llm.model.layers)
    if not 0 <= args.premature_layer < n_layers:
        raise ValueError(f"--premature_layer {args.premature_layer} out of range "
                         f"for a {n_layers}-layer LLM")
    print(f"[dola] LLM layers={n_layers}, premature=L{args.premature_layer} "
          f"({100*args.premature_layer/n_layers:.0f}% depth)", flush=True)

    capture = PrematureLayerCapture(model, [args.premature_layer]).register()
    step_fn = make_dola_step_fn(capture, args.premature_layer,
                                alpha=args.alpha, apc_beta=args.apc_beta)
    terminators = get_terminators(model, tokenizer)

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

            msgs = [{'role': 'user', 'content': content}]

            if is_omni:
                # Omni mode: use model's built-in omni system prompt
                msgs.insert(0, model.get_sys_prompt(mode='omni', language='en'))
            elif system_prompt:
                msgs.insert(0, {'role': 'system', 'content': [system_prompt]})

            # Same preprocessing as chat(); only the decoding differs.
            inputs = build_minicpm_inputs(
                model, model.processor, msgs,
                omni_input=is_omni, max_slice_nums=1, use_image_id=False)
            embeds = get_multimodal_embeds(model, inputs)

            capture.clear()
            response, _ = greedy_generate(
                model, tokenizer, embeds,
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=args.max_new_tokens,
                step_fn=step_fn,
                repetition_penalty=args.repetition_penalty,
                terminators=terminators,
            )

            elapsed = time.time() - t0
            print(f'  response[:120]: {response[:120]!r}', flush=True)
            print(f'  time={elapsed:.1f}s', flush=True)

        except Exception as e:
            print(f'  [ERROR] {e}', flush=True)
            response = ''
            elapsed = time.time() - t0

        torch.cuda.empty_cache()

        result = make_result(item, response, args.datasets, 'minicpm_dola',
                             time_sec=elapsed)
        save_result(result, output_path)

    total_time = time.time() - start
    n_done = len(data) - len(done_ids)
    print(f'\n[done] {n_done} samples in {total_time/60:.1f} min. Results: {output_path}', flush=True)

    capture.remove()

    summary = os.path.join(args.time_path, args.file_name + '.txt')
    with open(summary, 'w') as f:
        f.write(f'total_time: {total_time:.2f}s\n')
        f.write(f'samples: {n_done}\n')
        f.write(f'premature_layer: {args.premature_layer}\n')
        f.write(f'alpha: {args.alpha}\n')
        f.write(f'apc_beta: {args.apc_beta}\n')
        f.write(f'repetition_penalty: {args.repetition_penalty}\n')
        f.write('decode: greedy, per-token DoLa contrast (bypasses model.chat)\n')


if __name__ == '__main__':
    main()
