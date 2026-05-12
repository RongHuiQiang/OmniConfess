# -*- coding: utf-8 -*-
"""
base_model.py — Single-pass greedy baseline for all 9 datasets.

Usage:
    python base_model.py --datasets cmm --data_path /path/to/cmm.json --dry_run
    python base_model.py --datasets halueval --data_path /path/to/halueval.json \
        --output_dir ./results --file_name base_model_halueval
"""
import os, sys, time, argparse, random, torch, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared import (
    load_dataset, load_media_for_item,
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
    make_result, save_result,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',      type=str, required=True,
                        help='Dataset name: cmm | phd | haloquest | avhbench | halueval | ragtruth | pubmedqa')
    parser.add_argument('--data_path',     type=str, required=True)
    parser.add_argument('--data_root',     type=str, default='./OmniHalluBench/')
    parser.add_argument('--output_dir',    type=str, default='./results')
    parser.add_argument('--file_name',     type=str, default='base_model_out')
    parser.add_argument('--time_path',     type=str, default='./results/time')
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument('--max_new_tokens',type=int, default=512)
    parser.add_argument('--prompt_version',type=str, default='v1', choices=['v0','v1','v2','v3'])
    parser.add_argument('--dry_run',       action='store_true',
                        help='Only process first 3 items (smoke test)')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)

    output_path = os.path.join(args.output_dir, args.file_name + '.jsonl')

    # ── Resume: skip already-done items ──────────────────────────────────────
    done_ids = set()
    if os.path.exists(output_path):
        import json
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line.strip())['id'])
                except Exception:
                    pass
        print(f'[resume] skipping {len(done_ids)} already-done samples', flush=True)

    # ── Load model ───────────────────────────────────────────────────────────
    print('[model] loading Qwen2.5-Omni...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer

    # ── Load dataset ─────────────────────────────────────────────────────────
    data = load_dataset(args.data_path, args.data_root)
    if args.dry_run:
        data = data[:3]
        print(f'[dry_run] processing {len(data)} samples', flush=True)

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)

    all_input_tokens = 0
    all_output_tokens = 0
    start = time.time()

    for idx, item in enumerate(data):
        if item['id'] in done_ids:
            continue

        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        # ── Media ────────────────────────────────────────────────────────────
        pixel_values, audio_values, image_values = load_media_for_item(
            item, args.data_path, args.data_root)

        # ── Prompt ───────────────────────────────────────────────────────────
        user_prompt = get_user_prompt(
            item['question'], args.datasets, args.prompt_version,
            passage=item.get('passage'))
        user_content = build_user_content_with_media_tokens(
            pixel_values, audio_values, image_values, user_prompt)
        messages = make_messages(system_prompt, user_content)

        # ── Build inputs ─────────────────────────────────────────────────────
        try:
            raw_inputs = build_inputs(
                processor, model, messages,
                pixel_values, audio_values, image_values, args.datasets)
            clean = prepare_for_thinker(raw_inputs, model)
            input_len = clean['input_ids'].shape[1]
            all_input_tokens += input_len

            # ── Generate ────────────────────────────────────────────────────
            with torch.inference_mode():
                out = model.thinker.generate(
                    **clean,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            gen_ids = out[0][input_len:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            all_output_tokens += len(gen_ids)

            elapsed = time.time() - t0
            print(f'  response[:120]: {response[:120]!r}', flush=True)
            print(f'  tokens_out={len(gen_ids)} | time={elapsed:.1f}s', flush=True)

        except Exception as e:
            print(f'  [ERROR] {e}', flush=True)
            response = ''
            elapsed = time.time() - t0

        torch.cuda.empty_cache()

        result = make_result(item, response, args.datasets, 'base_model',
                             time_sec=elapsed)
        save_result(result, output_path)

    # ── Summary ──────────────────────────────────────────────────────────────
    total_time = time.time() - start
    summary = os.path.join(args.time_path, args.file_name + '.txt')
    with open(summary, 'w') as f:
        f.write(f'total_time: {total_time:.2f}s\n')
        f.write(f'total_input_tokens: {all_input_tokens}\n')
        f.write(f'total_output_tokens: {all_output_tokens}\n')
    print(f'\n[done] total={total_time:.1f}s | output={output_path}', flush=True)


if __name__ == '__main__':
    main()
