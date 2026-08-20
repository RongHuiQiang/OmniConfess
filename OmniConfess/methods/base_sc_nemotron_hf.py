# -*- coding: utf-8 -*-
"""
base_sc_nemotron_hf.py -- Self-Consistency baseline on NVIDIA
Nemotron-3-Nano-Omni-30B (transformers trust-remote-code path).

SC = draw `num_rollout` independent samples (do_sample=True, temperature) and
majority-vote the parsed predictions. This is a decode-strategy-only method: it
does not touch the model's internal structure, so it is architecture-agnostic
and runs on the hybrid Mamba/Attention Nemotron LLM exactly as on any other
CausalLM.

Structural mirror of base_model_nemotron_hf_pv.py (same media loading, prompt
building, processor call, buffer repair) + base_sc_qwen3.py (sampling + vote).

Trajectory recording is DISABLED for the Nemotron arm (no --record_process);
only the final prediction + all raw responses (as cot) are stored.

Usage:
  CUDA_VISIBLE_DEVICES=0 $PY base_sc_nemotron_hf.py \
      --datasets pubmedqa \
      --data_path OmniHalluBench/pubmedqa/pubmedqa.json \
      --data_root $PWD/OmniHalluBench/pubmedqa/ \
      --output_dir ./outputs/pubmedqa_nemotron_sc \
      --file_name pubmedqa_nemotron_sc --dry_run
"""
import os, sys, time, json, argparse, random
from collections import Counter
import numpy as np
import torch

REPO = os.environ.get('NEMO_REPO', '/home/tsq/rhq/omniconfess/OmniConfess')
sys.path.insert(0, os.path.join(REPO, 'OmniConfess', 'methods'))
from shared import (
    load_dataset, resolve_media_path,
    get_system_prompt, get_user_prompt,
    make_result, save_result, parse_pred,
)
from shared.data_loader import load_image
from shared.nemotron_shim import load_nemotron

# reuse the media loaders from the base runner (native-res video, 16k audio)
from base_model_nemotron_hf_pv import (
    load_video_frames_native, load_audio_wave, set_seed,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def majority_vote(predictions):
    if not predictions:
        return ''
    return Counter(predictions).most_common(1)[0][0]


def build_processor_inputs(processor, tokenizer, item, args, system_prompt, device):
    """Build the multimodal processor inputs for one item (media + prompt).
    Returns (inputs_dict_on_device, input_len, media_flags) or raises."""
    json_dir = os.path.dirname(args.data_path)
    videos = images = audio = None
    has_video = has_audio = has_image = False
    if item.get('video_path'):
        vp = resolve_media_path(item['video_path'], json_dir, args.data_root)
        videos = load_video_frames_native(vp)
        has_video = videos is not None
    if item.get('audio_path'):
        ap = resolve_media_path(item['audio_path'], json_dir, args.data_root)
        audio = load_audio_wave(ap)
        has_audio = audio is not None
    if item.get('image_path'):
        ip = resolve_media_path(item['image_path'], json_dir, args.data_root)
        images = load_image(ip)
        has_image = images is not None

    user_prompt = get_user_prompt(
        item['question'], args.datasets, args.prompt_version,
        passage=item.get('passage'))
    media_prefix = ''
    if has_video:
        media_prefix += '<video>\n'
    if has_audio:
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
    inputs = processor(**proc_kwargs)
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in inputs.items()}
    for _meta_key in ('num_patches', 'num_tokens', 'imgs_sizes'):
        inputs.pop(_meta_key, None)
    input_len = int(inputs['input_ids'].shape[1])
    return inputs, input_len, (has_video, has_audio, has_image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',      type=str, required=True)
    parser.add_argument('--data_path',     type=str, required=True)
    parser.add_argument('--data_root',     type=str, default=None)
    parser.add_argument('--output_dir',    type=str, default='./outputs')
    parser.add_argument('--file_name',     type=str, default='base_sc_nemotron')
    parser.add_argument('--time_path',     type=str, default='./outputs/time')
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument('--num_rollout',   type=int, default=5)
    parser.add_argument('--max_new_tokens',type=int, default=512)
    parser.add_argument('--temperature',   type=float, default=0.7)
    parser.add_argument('--top_p',         type=float, default=0.9)
    parser.add_argument('--prompt_version',type=str, default='v1',
                        choices=['v0','v1','v2','v3','v4','v5','v6'])
    parser.add_argument('--model_path',    type=str,
                        default=os.environ.get('NEMOTRON_MODEL_PATH'))
    parser.add_argument('--enable_thinking', action='store_true')
    parser.add_argument('--max_samples',   type=int, default=-1)
    parser.add_argument('--dry_run',       action='store_true')
    args = parser.parse_args()
    if not args.data_root:
        args.data_root = os.path.join(REPO, 'OmniHalluBench', args.datasets) + os.sep

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.file_name + '.jsonl')

    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line.strip())['id'])
                except Exception:
                    pass
        print(f'[resume] skipping {len(done_ids)} done samples', flush=True)

    processor, model, tokenizer, DEVICE = load_nemotron(args.model_path)
    print(f'[sc] num_rollout={args.num_rollout} temp={args.temperature} '
          f'top_p={args.top_p}', flush=True)

    data = load_dataset(args.data_path, args.data_root)
    if args.dry_run:
        data = data[:3]
        print(f'[dry_run] {len(data)} samples', flush=True)
    elif args.max_samples > 0:
        data = data[:args.max_samples]

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)
    all_in = all_out = 0
    start = time.time()

    for idx, item in enumerate(data):
        if item['id'] in done_ids:
            continue
        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        all_responses = []
        final_pred = ''
        input_len = 0
        media = (False, False, False)
        try:
            inputs, input_len, media = build_processor_inputs(
                processor, tokenizer, item, args, system_prompt, DEVICE)
            all_in += input_len * args.num_rollout

            for s in range(args.num_rollout):
                gen_kwargs = dict(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id or 0,
                )
                with torch.inference_mode():
                    out = model.generate(**inputs, **gen_kwargs)
                seq = out.sequences[0] if hasattr(out, 'sequences') else out[0]
                gen_ids = seq[input_len:]
                response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                all_responses.append(response)
                all_out += int(len(gen_ids))
                print(f'  rollout {s+1}/{args.num_rollout} -> {response[:80]!r}', flush=True)
                if not os.environ.get('NEMO_KEEP_CACHE'):
                    torch.cuda.empty_cache()

            all_preds = [parse_pred(r, args.datasets) for r in all_responses]
            final_pred = majority_vote(all_preds)
            print(f'  votes={Counter(all_preds)} -> final={final_pred!r}', flush=True)
        except Exception as e:
            import traceback
            print(f'  [ERROR] {e}', flush=True)
            traceback.print_exc()
            all_responses = []
            final_pred = ''

        elapsed = time.time() - t0
        if not os.environ.get('NEMO_KEEP_CACHE'):
            torch.cuda.empty_cache()

        result = make_result(item, all_responses[0] if all_responses else '',
                             args.datasets, 'base_sc_nemotron', time_sec=elapsed,
                             cot=json.dumps(all_responses, ensure_ascii=False))
        result['pred'] = final_pred
        result['nemotron'] = {
            'runtime': 'transformers-trust-remote-code',
            'method': 'sc', 'num_rollout': args.num_rollout,
            'temperature': args.temperature, 'top_p': args.top_p,
            'enable_thinking': bool(args.enable_thinking),
            'media': {'video': media[0], 'audio': media[1], 'image': media[2]},
        }
        save_result(result, output_path)

    total_time = time.time() - start
    with open(os.path.join(args.time_path, args.file_name + '.txt'), 'w') as f:
        f.write(f'total_time: {total_time:.2f}s\n')
        f.write(f'num_rollout: {args.num_rollout}\n')
        f.write(f'total_input_tokens: {all_in}\n')
        f.write(f'total_output_tokens: {all_out}\n')
    print(f'\n[done] total={total_time:.1f}s | output={output_path}', flush=True)


if __name__ == '__main__':
    main()
