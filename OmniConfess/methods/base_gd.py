# -*- coding: utf-8 -*-
"""
base_gd.py — Guided Decoding: step-level generation + judge-weighted beam search.

Algorithm:
  For each foresight step T:
    1. Generate num_rollout candidate steps per beam (sampling, 128 tokens)
    2. Judge each candidate: "Is this step correct? (A)/(B)" -> judge_p
    3. weight = sqrt(gen_prob) * sqrt(judge_p)
    4. Softmax resample -> keep step_beam_size paths
  Final: complete top beam, decode answer.

Usage:
    python base_gd.py --datasets cmm --data_path /path/to/cmm.json --dry_run
"""
import os, sys, time, argparse, random, json
import torch, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared import (
    load_dataset, load_media_for_item,
    load_model, build_inputs, prepare_for_thinker,
    build_user_content_with_media_tokens, make_messages,
    get_system_prompt, get_user_prompt,
    make_result, save_result,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

GD_JUDGE_PROMPT = (
    "Is the following reasoning step logically correct and consistent with the "
    "evidence in the media? Answer ONLY with (A) Correct or (B) Incorrect."
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def softmax(x):
    x = np.array(x, dtype=float)
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',       type=str, required=True)
    parser.add_argument('--data_path',      type=str, required=True)
    parser.add_argument('--data_root',      type=str, default='./OmniHalluBench/')
    parser.add_argument('--output_dir',     type=str, default='./results')
    parser.add_argument('--file_name',      type=str, default='base_gd_out')
    parser.add_argument('--time_path',      type=str, default='./results/time')
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--step_beam_size', type=int, default=2)
    parser.add_argument('--num_rollout',    type=int, default=4)
    parser.add_argument('--num_foresight',  type=int, default=3)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--prompt_version', type=str, default='v1', choices=['v0','v1','v2','v3'])
    parser.add_argument('--dry_run',        action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)

    output_path = os.path.join(args.output_dir, args.file_name + '.jsonl')

    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line.strip())['id'])
                except Exception:
                    pass

    print('[model] loading...', flush=True)
    processor, model = load_model()
    tokenizer = processor.tokenizer

    data = load_dataset(args.data_path, args.data_root)
    if args.dry_run:
        data = data[:3]
        print(f'[dry_run] {len(data)} samples', flush=True)

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)
    start = time.time()
    all_input_tokens = 0
    all_output_tokens = 0

    for idx, item in enumerate(data):
        if item['id'] in done_ids:
            continue

        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        pixel_values, audio_values, image_values = load_media_for_item(
            item, args.data_path, args.data_root)

        q_text = item['question']
        passage = item.get('passage')

        # ── Beam state ───────────────────────────────────────────────────────
        previous_steps = ['The reasoning steps are:\n\n'] * args.step_beam_size
        # Track per-beam cumulative logp lists and token-length lists
        steps_cumlogp = [[] for _ in range(args.step_beam_size)]
        steps_len = [[] for _ in range(args.step_beam_size)]
        try_count = 0
        _t_gen = 0.0
        _t_sel = 0.0

        while try_count < 3:
          try:
            for T in range(args.num_foresight):
                tem_steps, tem_logps, tem_lens, tem_judge_p, tem_parent = [], [], [], [], []

                for beam_idx in range(args.step_beam_size):
                    # ── Build step-expansion prompt ──────────────────────────
                    base_prompt = get_user_prompt(q_text, args.datasets,
                                                  args.prompt_version, passage=passage)
                    step_prompt = (
                        f"Problem: {q_text}\n"
                        f"{previous_steps[beam_idx]}\n"
                        "Please generate the NEXT logical reasoning step. Do NOT provide the final answer yet. Focus strictly on analyzing the physical details step by step."
                    )
                    user_content = build_user_content_with_media_tokens(
                        pixel_values, audio_values, image_values, step_prompt)
                    messages = make_messages(system_prompt, user_content)

                    raw_inp = build_inputs(
                        processor, model, messages,
                        pixel_values, audio_values, image_values, args.datasets)
                    clean = prepare_for_thinker(raw_inp, model)
                    input_len = clean['input_ids'].shape[1]
                    all_input_tokens += input_len

                    # ── Generate num_rollout candidate steps ──────────────────
                    _tg0 = time.perf_counter()
                    with torch.no_grad():
                        rollout_out = model.thinker.generate(
                            **clean,
                            max_new_tokens=128,
                            do_sample=True,
                            temperature=0.6,
                            num_return_sequences=args.num_rollout,
                            output_scores=True,
                            return_dict_in_generate=True,
                            pad_token_id=tokenizer.eos_token_id,
                            stop_strings=['\n\n', '<end_of_reasoning>'],
                            tokenizer=tokenizer,
                        )
                        trans_scores = model.thinker.compute_transition_scores(
                            rollout_out.sequences, rollout_out.scores, normalize_logits=True)
                        path_logps = torch.mean(trans_scores, dim=1).cpu().numpy()

                    for j in range(args.num_rollout):
                        gen_ids = rollout_out.sequences[j][input_len:]
                        gen_ids = gen_ids[gen_ids != tokenizer.pad_token_id]
                        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                        all_output_tokens += len(gen_ids)
                        tem_steps.append(response)
                        tem_logps.append(float(path_logps[j]))
                        tem_lens.append(len(gen_ids))
                        tem_parent.append(beam_idx)

                    torch.cuda.empty_cache()
                    _t_gen += time.perf_counter() - _tg0

                # ── Judge each candidate ─────────────────────────────────────
                _ts0 = time.perf_counter()
                for cidx, step_text in enumerate(tem_steps):
                    parent_idx = tem_parent[cidx]
                    eval_msg = (
                        f"Problem: {q_text}\n"
                        f"Previous reasoning:\n{previous_steps[parent_idx]}\n"
                        f"Current step: {step_text}\n"
                        f"{GD_JUDGE_PROMPT}"
                    )
                    eval_content = build_user_content_with_media_tokens(
                        pixel_values, audio_values, image_values, eval_msg)
                    eval_messages = make_messages(system_prompt, eval_content)

                    raw_eval = build_inputs(
                        processor, model, eval_messages,
                        pixel_values, audio_values, image_values, args.datasets)
                    clean_eval = prepare_for_thinker(raw_eval, model)

                    with torch.no_grad():
                        e_out = model.thinker.generate(
                            **clean_eval,
                            max_new_tokens=5,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    e_res = tokenizer.decode(
                        e_out[0][clean_eval['input_ids'].shape[1]:],
                        skip_special_tokens=True)
                    judge_p = 0.95 if '(A)' in e_res or 'Correct' in e_res else 0.05
                    tem_judge_p.append(judge_p)
                    torch.cuda.empty_cache()

                # ── Weight and resample ──────────────────────────────────────
                weights = []
                for cidx in range(len(tem_steps)):
                    p_idx = tem_parent[cidx]
                    hist_logp = sum(steps_cumlogp[p_idx]) if steps_cumlogp[p_idx] else 0.0
                    hist_len  = sum(steps_len[p_idx])     if steps_len[p_idx]     else 0
                    total_avg = (hist_logp + tem_logps[cidx] * tem_lens[cidx]) / (
                        hist_len + tem_lens[cidx] + 1e-8)
                    gen_prob  = np.exp(total_avg)
                    w = (gen_prob ** 0.5) * (tem_judge_p[cidx] ** 0.5)
                    weights.append(w)

                norm_w = softmax(weights)
                selected = np.random.choice(
                    len(norm_w), p=norm_w, size=args.step_beam_size, replace=True).tolist()

                new_steps   = []
                new_cumlogp = []
                new_lens    = []
                for sel_idx in selected:
                    p_idx = tem_parent[sel_idx]
                    new_steps.append(previous_steps[p_idx] + tem_steps[sel_idx] + '\n')
                    new_cumlogp.append(steps_cumlogp[p_idx] + [tem_logps[sel_idx]])
                    new_lens.append(steps_len[p_idx] + [tem_lens[sel_idx]])

                previous_steps  = new_steps
                steps_cumlogp   = new_cumlogp
                steps_len       = new_lens
                _t_sel += time.perf_counter() - _ts0
                print(f'  step {T+1}/{args.num_foresight} done', flush=True)

            # ── Final completion: complete ALL beams, then judge-select best ──
            _tg1 = time.perf_counter()
            beam_responses = []
            beam_trajs    = []

            for beam_idx in range(len(previous_steps)):
                final_step_prompt = (
                    f"Question: {q_text}\n"
                    f"{previous_steps[beam_idx]}\n"
                    "Please provide the final answer based on the reasoning above:"
                )
                final_content = build_user_content_with_media_tokens(
                    pixel_values, audio_values, image_values, final_step_prompt)
                final_messages = make_messages(system_prompt, final_content)

                raw_final = build_inputs(
                    processor, model, final_messages,
                    pixel_values, audio_values, image_values, args.datasets)
                clean_final = prepare_for_thinker(raw_final, model)
                final_input_len = clean_final['input_ids'].shape[1]
                all_input_tokens += final_input_len

                with torch.inference_mode():
                    f_out = model.thinker.generate(
                        **clean_final,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        output_scores=True,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                f_gen_ids = f_out.sequences[0][final_input_len:]
                valid_mask = (f_gen_ids != tokenizer.pad_token_id)
                f_gen_ids = f_gen_ids[valid_mask]
                f_response = tokenizer.decode(f_gen_ids, skip_special_tokens=True).strip()
                all_output_tokens += len(f_gen_ids)
                beam_responses.append(f_response)
                beam_trajs.append(previous_steps[beam_idx] + '\n' + f_response)
                torch.cuda.empty_cache()

            _t_gen += time.perf_counter() - _tg1

            # ── Judge each completed answer, then pick best beam ─────────────
            _ts1 = time.perf_counter()
            tem_judge_p_list = []
            for ijdx, candidate in enumerate(beam_responses):
                eval_final_msg = (
                    f"Problem: {q_text}\n"
                    f"Final Reasoning and Answer: {candidate}\n"
                    "Is this final answer correct based on the media evidence? "
                    "Answer (A) Correct or (B) Incorrect."
                )
                eval_content = build_user_content_with_media_tokens(
                    pixel_values, audio_values, image_values, eval_final_msg)
                eval_messages = make_messages(system_prompt, eval_content)
                raw_eval = build_inputs(
                    processor, model, eval_messages,
                    pixel_values, audio_values, image_values, args.datasets)
                clean_eval = prepare_for_thinker(raw_eval, model)
                with torch.no_grad():
                    e_out = model.thinker.generate(
                        **clean_eval,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                e_res = tokenizer.decode(
                    e_out[0][clean_eval['input_ids'].shape[1]:],
                    skip_special_tokens=True)
                judge_p = 0.95 if '(A)' in e_res or 'Correct' in e_res else 0.05
                tem_judge_p_list.append(judge_p)
                torch.cuda.empty_cache()

            # Weighted selection: exp(mean_cumlogp)^0.5 * judge_p^0.5
            weights = []
            for jidx in range(len(previous_steps)):
                hist_logp = sum(steps_cumlogp[jidx]) if steps_cumlogp[jidx] else 0.0
                hist_len  = sum(steps_len[jidx])     if steps_len[jidx]     else 0
                mean_logp = hist_logp / (hist_len + 1e-8)
                gen_p     = np.exp(mean_logp) ** 0.5
                j_p       = (tem_judge_p_list[jidx]) ** 0.5
                weights.append(gen_p * j_p)

            norm_w = softmax(np.array(weights))
            best_beam = int(np.random.choice(len(norm_w), p=norm_w))
            print(f'  final judge weights={[round(w,3) for w in weights]} | best_beam={best_beam}',
                  flush=True)

            final_response = beam_responses[best_beam]
            cot = beam_trajs[best_beam]
            _t_sel += time.perf_counter() - _ts1

            break   # success

          except Exception as e:
            try_count += 1
            print(f'  [ERROR attempt {try_count}/3] {e}', flush=True)
            torch.cuda.empty_cache()
            if try_count == 3:
                final_response = ''
                cot = ''

        elapsed = time.time() - t0
        torch.cuda.empty_cache()
        print(f'  response[:100]: {final_response[:100]!r} | time={elapsed:.1f}s', flush=True)

        result = make_result(item, final_response, args.datasets, 'base_gd',
                             time_sec=elapsed, cot=cot)
        result['generation_time_seconds'] = round(_t_gen, 4)
        result['selection_time_seconds'] = round(_t_sel, 4)
        save_result(result, output_path)

    total_time = time.time() - start
    summary = os.path.join(args.time_path, args.file_name + '.txt')
    with open(summary, 'w') as f:
        f.write(f'total_time: {total_time:.2f}s\n')
        f.write(f'step_beam_size: {args.step_beam_size}\n')
        f.write(f'num_rollout: {args.num_rollout}\n')
        f.write(f'num_foresight: {args.num_foresight}\n')
        f.write(f'total_input_tokens: {all_input_tokens}\n')
        f.write(f'total_output_tokens: {all_output_tokens}\n')
    print(f'\n[done] {total_time:.1f}s | {output_path}', flush=True)


if __name__ == '__main__':
    main()
