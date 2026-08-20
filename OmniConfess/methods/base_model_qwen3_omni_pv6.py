# -*- coding: utf-8 -*-
"""
base_model_qwen3_omni.py — Single-pass greedy baseline on Qwen3-Omni-30B-A3B-Instruct
(third true-omnimodal backbone: image+video+audio+text).

Structural mirror of methods/base_model.py (Qwen2.5-Omni arm): same shared v1
prompts, same data_loader, same result schema, same --record_process trajectory
(phi-Decoding style). Only the model layer differs -- see
shared/qwen3_omni_utils.py for the Qwen3-Omni deltas (token spellings,
pixel_values key, thinker-only loading).

Usage (one process per card; the thinker alone is ~62GB bf16):

NOTE (2026-08-16): eos_token_id is passed explicitly in gen_kwargs -- see the
comment near load_model(). Do not remove it.
    CUDA_VISIBLE_DEVICES=0 python base_model_qwen3_omni.py \
        --datasets pubmedqa --data_path ./OmniHalluBench/pubmedqa/pubmedqa.json \
        --output_dir ./outputs/pubmedqa_qwen3omni --file_name base_model_qwen3_omni_shard0 \
        --dry_run
"""
import os, sys, time, argparse, random, torch, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared import (
    load_dataset, load_media_for_item,
    get_system_prompt, get_user_prompt,
    make_result, save_result,
)
from shared.qwen3_omni_utils import (
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
)


def token_logprobs(gen_ids, scores):
    """Per-token logprob of each generated token from the greedy-step scores.

    Mirrors the phi-Decoding Baseline-GD.py recording style (identical to
    base_model.py) so downstream analysis is backbone-agnostic.
    """
    out = []
    for tid, logits in zip(gen_ids.tolist(), scores):
        logits = logits.float()
        if logits.dim() > 1:          # scores come back as [batch, vocab]
            logits = logits[0]
        lp = torch.log_softmax(logits, dim=-1)[tid]
        out.append(round(lp.item(), 4))
    return out


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',      type=str, required=True,
                        help='cmm | phd | haloquest | avhbench | ragtruth | pubmedqa')
    parser.add_argument('--data_path',     type=str, required=True)
    parser.add_argument('--data_root',     type=str, default='./OmniHalluBench/')
    parser.add_argument('--output_dir',    type=str, default='./outputs')
    parser.add_argument('--file_name',     type=str, default='base_model_qwen3_omni')
    parser.add_argument('--time_path',     type=str, default='./outputs/time')
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument('--max_new_tokens',type=int, default=512)
    parser.add_argument('--prompt_version',type=str, default='v1',
                        choices=['v0','v1','v2','v3','v4','v5','v6'])
    parser.add_argument('--record_process', action=argparse.BooleanOptionalAction, default=True,
                        help='Record per-token trajectory (token ids, per-token logprobs, '
                             'cumulative/norm logp) alongside the response')
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
    print('[model] loading Qwen3-Omni-30B-A3B-Instruct thinker...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer

    # Qwen3-Omni's generation_config.json ships WITHOUT eos_token_id (only
    # talker_* keys), so model.generate() never stops early -- every response
    # runs to the max_new_tokens cap and appends degenerate chatter after
    # <|im_end|> (first avhbench arm: 7.55 Token-F1, 87s/item). This is the
    # eos fix back-written from the GPFS copy (2026-08-16, RESULTS.md pending
    # item): stop ids must be passed explicitly. (pad_token_id never triggers
    # stopping; it only masks.)
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    endoftext_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
    eos_ids = [i for i in {im_end_id, endoftext_id} if i is not None]

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
        has_audio = audio_values is not None
        has_video = pixel_values is not None
        has_image = image_values is not None

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
            clean = prepare_for_thinker(raw_inputs)
            input_len = clean['input_ids'].shape[1]
            all_input_tokens += input_len

            if has_audio:
                n_feat = int(raw_inputs['input_features'].shape[-1]) if 'input_features' in raw_inputs else 0
                print(f'  [audio] input_features={n_feat} (non-empty => audio really in)', flush=True)

            # ── Generate ────────────────────────────────────────────────────
            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=eos_ids,
                pad_token_id=endoftext_id,
            )
            if args.record_process:
                gen_kwargs.update(output_scores=True,
                                  return_dict_in_generate=True)
            with torch.inference_mode():
                out = model.generate(**clean, **gen_kwargs)

            if args.record_process:
                gen_ids = out.sequences[0][input_len:]
                logp_list = token_logprobs(gen_ids.cpu(), out.scores)
            else:
                gen_ids = out[0][input_len:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            all_output_tokens += len(gen_ids)

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

        torch.cuda.empty_cache()

        result = make_result(item, response, args.datasets, 'base_model_qwen3_omni',
                             time_sec=elapsed)

        # ── Trajectory record (phi-Decoding Baseline-GD.py style) ───────────
        if args.record_process:
            result['record_process'] = True
            if gen_ids is not None:
                cum_lp = float(np.sum(logp_list))
                result['token_ids'] = gen_ids.tolist()
                result['logp_list'] = logp_list
                result['cumulative_logprob'] = round(cum_lp, 4)
                result['norm_logp'] = round(cum_lp / (len(gen_ids) + 1e-8), 4)
                result['input_len'] = int(input_len)
                result['output_tokens'] = int(len(gen_ids))
            else:
                result['token_ids'] = None
                result['logp_list'] = None
                result['cumulative_logprob'] = None
                result['norm_logp'] = None
                result['input_len'] = None
                result['output_tokens'] = 0

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
