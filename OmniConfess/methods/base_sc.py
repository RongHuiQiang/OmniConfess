# -*- coding: utf-8 -*-
"""
base_sc.py — Self-Consistency: serial independent sampling + majority vote.

Usage:
    python base_sc.py --datasets cmm --data_path /path/to/cmm.json --dry_run
"""
import os, sys, time, argparse, random, torch, numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared import (
    load_dataset, load_media_for_item,
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
    parse_pred, make_result, save_result,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def majority_vote(predictions):
    """Return most common prediction; tie-break by first occurrence."""
    if not predictions:
        return ''
    return Counter(predictions).most_common(1)[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',      type=str, required=True)
    parser.add_argument('--data_path',     type=str, required=True)
    parser.add_argument('--data_root',     type=str, default='./OmniHalluBench/')
    parser.add_argument('--output_dir',    type=str, default='./results')
    parser.add_argument('--file_name',     type=str, default='base_sc_out')
    parser.add_argument('--time_path',     type=str, default='./results/time')
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument('--num_rollout',   type=int, default=4,
                        help='Number of independent sampling paths')
    parser.add_argument('--max_new_tokens',type=int, default=512)
    parser.add_argument('--temperature',   type=float, default=0.6)
    parser.add_argument('--prompt_version',type=str, default='v1', choices=['v0','v1','v2','v3'])
    parser.add_argument('--dry_run',       action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)

    output_path = os.path.join(args.output_dir, args.file_name + '.jsonl')

    done_ids = set()
    if os.path.exists(output_path):
        import json
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line.strip())['id'])
                except Exception:
                    pass
        print(f'[resume] skipping {len(done_ids)} samples', flush=True)

    print('[model] loading...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer

    data = load_dataset(args.data_path, args.data_root)
    if args.dry_run:
        data = data[:3]
        print(f'[dry_run] {len(data)} samples', flush=True)

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)
    all_input_tokens = 0
    all_output_tokens = 0
    start = time.time()

    for idx, item in enumerate(data):
        if item['id'] in done_ids:
            continue

        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        pixel_values, audio_values, image_values = load_media_for_item(
            item, args.data_path, args.data_root)

        base_prompt = get_user_prompt(
            item['question'], args.datasets, args.prompt_version,
            passage=item.get('passage'))
            
        user_prompt = base_prompt + "\nPlease think step by step, and then provide your final answer."
        
        user_content = build_user_content_with_media_tokens(
            pixel_values, audio_values, image_values, user_prompt)
        messages = make_messages(system_prompt, user_content)

        all_responses = []
        final_pred = ''
        try_count = 0
        _t_gen = 0.0
        _t_sel = 0.0

        while try_count < 3:
            try:
                raw_inputs = build_inputs(
                    processor, model, messages,
                    pixel_values, audio_values, image_values, args.datasets)
                clean = prepare_for_thinker(raw_inputs, model)
                input_len = clean['input_ids'].shape[1]
                all_input_tokens += input_len * args.num_rollout

                # ── Serial sampling ─────────────────────────────────────────
                all_responses = []
                _t_gen_start = time.perf_counter()
                for s in range(args.num_rollout):
                    print(f'  rollout {s+1}/{args.num_rollout}', flush=True)
                    with torch.inference_mode():
                        out = model.thinker.generate(
                            **clean,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=True,
                            temperature=args.temperature,
                            pad_token_id=tokenizer.eos_token_id,
                            stop_strings=['<end_of_reasoning>'],
                            tokenizer=tokenizer,
                        )
                    gen_ids = out[0][input_len:]
                    gen_ids = gen_ids[gen_ids != tokenizer.pad_token_id]
                    response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    all_responses.append(response)
                    all_output_tokens += len(gen_ids)
                    torch.cuda.empty_cache()
                _t_gen = time.perf_counter() - _t_gen_start

                # ── Majority vote ────────────────────────────────────────────
                _t_sel_start = time.perf_counter()
                all_preds = [parse_pred(r, args.datasets) for r in all_responses]
                final_pred = majority_vote(all_preds)
                _t_sel = time.perf_counter() - _t_sel_start
                print(f'  votes={Counter(all_preds)} -> final={final_pred!r}', flush=True)
                break

            except Exception as e:
                try_count += 1
                print(f'  [ERROR attempt {try_count}/3] {e}', flush=True)
                torch.cuda.empty_cache()
                if try_count == 3:
                    all_responses = []
                    final_pred = ''

        elapsed = time.time() - t0
        torch.cuda.empty_cache()

        # Build result: use final_pred as pred, all responses as cot
        import json as _json
        result = make_result(item, all_responses[0] if all_responses else '',
                             args.datasets, 'base_sc', time_sec=elapsed,
                             cot=_json.dumps(all_responses, ensure_ascii=False))
        # Override pred with majority-vote result
        result['pred'] = final_pred
        result['generation_time_seconds'] = round(_t_gen, 4)
        result['selection_time_seconds'] = round(_t_sel, 4)
        save_result(result, output_path)

    total_time = time.time() - start
    summary = os.path.join(args.time_path, args.file_name + '.txt')
    with open(summary, 'w') as f:
        f.write(f'total_time: {total_time:.2f}s\n')
        f.write(f'num_rollout: {args.num_rollout}\n')
        f.write(f'total_input_tokens: {all_input_tokens}\n')
        f.write(f'total_output_tokens: {all_output_tokens}\n')
    print(f'\n[done] {total_time:.1f}s | {output_path}', flush=True)


if __name__ == '__main__':
    main()
