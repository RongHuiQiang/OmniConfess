# -*- coding: utf-8 -*-
"""
base_model_qwen3_omni_var.py — Qwen3-Omni *input-perturbation* arms for the
Vote-SC cross-generation pool extension (2026-08-16, EXPERIMENT_DESIGN.md §6).

Exact copy of shared/base_model_qwen3_omni.py (the eos_token_id-fixed runner
that produced avhbench_qwen3omni = 19.03), with two provenance additions so
arms can be told apart in one pool without relying on directory names:

  result['model']  = --model_tag   (e.g. 'Qwen3-Omni-30B-A3B-Instruct')
  result['input_variant'] = --variant   (e.g. 'res384' | 'frames16' | 'base')
  result['frame_size'] / result['num_frames'] = the effective env-knob values

The perturbation itself is NOT in this file: shared/data_loader reads
OMNI_FRAME_SIZE / OMNI_NUM_FRAMES (defaults 224 / 8), and the launcher
(run_q3_perturb_avh.sh) exports them per arm. v1 prompt + greedy, identical
to the shipped q3 bare arm.

Usage (one process per card; thinker-only weights ~62GB bf16):
    CUDA_VISIBLE_DEVICES=0 python base_model_qwen3_omni_var.py \
        --datasets avhbench --data_path .../avhbench_shard0of8.json \
        --data_root ./OmniHalluBench/avhbench/ \
        --output_dir outputs/avhbench_vote_sc/q3_res384 \
        --file_name avhbench_q3_res384_shard0 \
        --model_tag Qwen3-Omni-30B-A3B-Instruct --variant res384
"""
import os, sys, time, argparse, random, torch, numpy as np

sys.path.insert(0, '/home/tsq/rhq/omniconfess/OmniConfess/OmniConfess/methods')
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
    """Per-token logprob of each generated token from the greedy-step scores."""
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
    parser.add_argument('--datasets',      type=str, required=True)
    parser.add_argument('--data_path',     type=str, required=True)
    parser.add_argument('--data_root',     type=str, default='./OmniHalluBench/')
    parser.add_argument('--output_dir',    type=str, default='./outputs')
    parser.add_argument('--file_name',     type=str, default='base_model_qwen3_omni_var')
    parser.add_argument('--time_path',     type=str, default='./outputs/time')
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument('--max_new_tokens',type=int, default=512)
    parser.add_argument('--prompt_version',type=str, default='v1', choices=['v0','v1','v2','v3'])
    parser.add_argument('--model_tag',     type=str, default='Qwen3-Omni-30B-A3B-Instruct')
    parser.add_argument('--variant',       type=str, default='base',
                        help='input-perturbation tag: base | res384 | frames16')
    parser.add_argument('--record_process', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--dry_run',       action='store_true')
    args = parser.parse_args()

    frame_size = int(os.environ.get("OMNI_FRAME_SIZE", "224"))
    num_frames = int(os.environ.get("OMNI_NUM_FRAMES", "8"))
    print(f'[variant] {args.variant} | OMNI_FRAME_SIZE={frame_size} '
          f'OMNI_NUM_FRAMES={num_frames}', flush=True)

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

    # Qwen3-Omni's generation_config.json has no eos_token_id -> generate()
    # would run to max_new_tokens every time. Stop ids from the tokenizer.
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

        pixel_values, audio_values, image_values = load_media_for_item(
            item, args.data_path, args.data_root)
        has_audio = audio_values is not None

        user_prompt = get_user_prompt(
            item['question'], args.datasets, args.prompt_version,
            passage=item.get('passage'))
        user_content = build_user_content_with_media_tokens(
            pixel_values, audio_values, image_values, user_prompt)
        messages = make_messages(system_prompt, user_content)

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
        # ── variant provenance (the only schema delta vs the bare q3 arm) ────
        result['model'] = args.model_tag
        result['input_variant'] = args.variant
        result['frame_size'] = frame_size
        result['num_frames'] = num_frames

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

    total_time = time.time() - start
    summary = os.path.join(args.time_path, args.file_name + '.txt')
    with open(summary, 'w') as f:
        f.write(f'total_time: {total_time:.2f}s\n')
        f.write(f'total_input_tokens: {all_input_tokens}\n')
        f.write(f'total_output_tokens: {all_output_tokens}\n')
    print(f'\n[done] total={total_time:.1f}s | output={output_path}', flush=True)


if __name__ == '__main__':
    main()
