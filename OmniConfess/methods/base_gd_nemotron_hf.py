# -*- coding: utf-8 -*-
"""
base_gd_nemotron_hf.py -- Guided Decoding baseline (step-level generation +
judge-weighted beam search) on Nemotron-3-Nano-Omni-30B.

Algorithm 1:1 from base_gd_qwen3.py (step_beam_size=2, num_rollout=4,
num_foresight=3, judge 0.95/0.05, NOT retuned): sample candidate steps, judge
each with a 5-token greedy probe, weight = sqrt(gen_prob)*sqrt(judge_p),
softmax-resample step_beam_size paths, final completion + judged beam choice.

Nemotron deltas vs the Qwen3 arm:
  * Rollouts are drawn SERIALLY (num_return_sequences batching not exercised
    on this wrapper); each rollout's avg per-token logp (the path score) comes
    from generate(output_scores=True) -- same quantity as
    compute_transition_scores(..., normalize_logits=True).mean(). Step length
    (used only to average path logps across steps) is the word count of the
    decoded step, as the raw token ids are not retained on this arm.
  * All generation via the outer omni wrapper's generate() (media fused by the
    wrapper); step '\n\n' stop applied post-hoc on the decoded text.
  * No trajectory recording (Nemotron arm policy).
"""
import os, sys, time, json, argparse, random
import numpy as np
import torch

REPO = os.environ.get('NEMO_REPO', '/home/tsq/rhq/omniconfess/OmniConfess')
sys.path.insert(0, os.path.join(REPO, 'OmniConfess', 'methods'))
from shared import load_dataset, get_system_prompt, make_result, save_result
from shared.nemotron_shim import load_nemotron
from shared.nemotron_methods import (
    load_item_media, inputs_from_text, outer_generate, set_seed,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

GD_JUDGE_PROMPT = (
    "Is the following reasoning step logically correct and consistent with the "
    "evidence in the media? Answer ONLY with (A) Correct or (B) Incorrect."
)


def softmax(x):
    x = np.array(x, dtype=float)
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',       type=str, required=True)
    parser.add_argument('--data_path',      type=str, required=True)
    parser.add_argument('--data_root',      type=str, default=None)
    parser.add_argument('--output_dir',     type=str, default='./outputs')
    parser.add_argument('--file_name',      type=str, default='gd_nemotron')
    parser.add_argument('--time_path',      type=str, default='./outputs/time')
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--step_beam_size', type=int, default=2)
    parser.add_argument('--num_rollout',    type=int, default=4)
    parser.add_argument('--num_foresight',  type=int, default=3)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--prompt_version', type=str, default='v1')
    parser.add_argument('--model_path',     type=str,
                        default=os.environ.get('NEMOTRON_MODEL_PATH'))
    parser.add_argument('--enable_thinking', action='store_true')
    parser.add_argument('--record_process', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--max_samples',    type=int, default=-1)
    parser.add_argument('--start_index',    type=int, default=0)
    parser.add_argument('--dry_run',        action='store_true')
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
    print(f'[gd] beam={args.step_beam_size} rollout={args.num_rollout} '
          f'foresight={args.num_foresight}', flush=True)

    data = load_dataset(args.data_path, args.data_root)
    if args.start_index > 0:
        data = data[args.start_index:]
    if args.dry_run:
        data = data[:3]
    elif args.max_samples > 0:
        data = data[:args.max_samples]

    system_prompt = get_system_prompt(args.datasets, args.prompt_version)
    start = time.time()
    all_in = all_out = 0

    def _gen(user_text, media, max_new_tokens=128, do_sample=True,
             temperature=0.6, stop_on_newline=False, want_logp=False):
        nonlocal all_in, all_out
        inputs, il = inputs_from_text(processor, tokenizer, system_prompt,
                                      user_text, media, DEVICE,
                                      enable_thinking=args.enable_thinking)
        all_in += il
        text, gen_ids, avg_lp = outer_generate(
            model, tokenizer, inputs, max_new_tokens=max_new_tokens,
            do_sample=do_sample, temperature=temperature, want_logp=want_logp)
        if stop_on_newline:
            text = text.split('\n\n')[0].strip()
        all_out += len(gen_ids)
        torch.cuda.empty_cache()
        return text, avg_lp

    with open(output_path, 'a', encoding='utf-8') as res_f:
        for idx, item in enumerate(data, start=args.start_index):
            item_id = item.get('id', idx)
            if item_id in done_ids:
                continue
            t0 = time.time()
            print(f'\n[{idx+1}/{len(data)}] id={item_id}', flush=True)
            media = load_item_media(item, args)
            q_text = item['question']

            previous_steps = ['The reasoning steps are:\n\n'] * args.step_beam_size
            steps_cumlogp = [[] for _ in range(args.step_beam_size)]
            steps_len = [[] for _ in range(args.step_beam_size)]
            final_response = ''
            cot = ''
            _t_gen = _t_sel = 0.0
            try_count = 0

            while try_count < 3:
              try:
                for T in range(args.num_foresight):
                    tem_steps, tem_logps, tem_lens, tem_judge_p, tem_parent = [], [], [], [], []
                    _tg0 = time.perf_counter()
                    for beam_idx in range(args.step_beam_size):
                        step_prompt = (
                            f"Problem: {q_text}\n"
                            f"{previous_steps[beam_idx]}\n"
                            "Please generate the NEXT logical reasoning step. Do NOT provide the final answer yet. Focus strictly on analyzing the physical details step by step.")
                        for j in range(args.num_rollout):
                            resp, avg_lp = _gen(step_prompt, media,
                                                max_new_tokens=128,
                                                do_sample=True, temperature=0.6,
                                                stop_on_newline=True,
                                                want_logp=True)
                            tem_steps.append(resp)
                            tem_logps.append(float(avg_lp) if avg_lp is not None else 0.0)
                            tem_lens.append(max(len(resp.split()), 1))
                            tem_parent.append(beam_idx)
                    _t_gen += time.perf_counter() - _tg0
                    torch.cuda.empty_cache()

                    _ts0 = time.perf_counter()
                    for cidx, step_text in enumerate(tem_steps):
                        parent_idx = tem_parent[cidx]
                        eval_msg = (
                            f"Problem: {q_text}\n"
                            f"Previous reasoning:\n{previous_steps[parent_idx]}\n"
                            f"Current step: {step_text}\n"
                            f"{GD_JUDGE_PROMPT}")
                        e_res, _ = _gen(eval_msg, media, max_new_tokens=5,
                                        do_sample=False)
                        tem_judge_p.append(0.95 if '(A)' in e_res or 'Correct' in e_res else 0.05)
                    _t_sel += time.perf_counter() - _ts0

                    weights = []
                    for cidx in range(len(tem_steps)):
                        p_idx = tem_parent[cidx]
                        hist_logp = sum(steps_cumlogp[p_idx]) if steps_cumlogp[p_idx] else 0.0
                        hist_len = sum(steps_len[p_idx]) if steps_len[p_idx] else 0
                        total_avg = (hist_logp + tem_logps[cidx] * tem_lens[cidx]) / (hist_len + tem_lens[cidx] + 1e-8)
                        gen_prob = np.exp(total_avg)
                        weights.append((gen_prob ** 0.5) * (tem_judge_p[cidx] ** 0.5))
                    norm_w = softmax(weights)
                    selected = np.random.choice(
                        len(norm_w), p=norm_w, size=args.step_beam_size,
                        replace=True).tolist()
                    new_steps, new_cumlogp, new_lens = [], [], []
                    for sel_idx in selected:
                        p_idx = tem_parent[sel_idx]
                        new_steps.append(previous_steps[p_idx] + tem_steps[sel_idx] + '\n')
                        new_cumlogp.append(steps_cumlogp[p_idx] + [tem_logps[sel_idx]])
                        new_lens.append(steps_len[p_idx] + [tem_lens[sel_idx]])
                    previous_steps, steps_cumlogp, steps_len = new_steps, new_cumlogp, new_lens
                    print(f'  step {T+1}/{args.num_foresight} done', flush=True)

                _tg1 = time.perf_counter()
                beam_responses, beam_trajs = [], []
                for beam_idx in range(len(previous_steps)):
                    final_step_prompt = (
                        f"Question: {q_text}\n"
                        f"{previous_steps[beam_idx]}\n"
                        "Please provide the final answer based on the reasoning above:")
                    f_response, _ = _gen(final_step_prompt, media,
                                         max_new_tokens=args.max_new_tokens,
                                         do_sample=True, temperature=0.7)
                    beam_responses.append(f_response)
                    beam_trajs.append(previous_steps[beam_idx] + '\n' + f_response)
                _t_gen += time.perf_counter() - _tg1

                _ts1 = time.perf_counter()
                tem_judge_p_list = []
                for candidate in beam_responses:
                    eval_final_msg = (
                        f"Problem: {q_text}\n"
                        f"Final Reasoning and Answer: {candidate}\n"
                        "Is this final answer correct based on the media evidence? "
                        "Answer (A) Correct or (B) Incorrect.")
                    e_res, _ = _gen(eval_final_msg, media, max_new_tokens=10,
                                    do_sample=False)
                    tem_judge_p_list.append(0.95 if '(A)' in e_res or 'Correct' in e_res else 0.05)

                weights = []
                for jidx in range(len(previous_steps)):
                    hist_logp = sum(steps_cumlogp[jidx]) if steps_cumlogp[jidx] else 0.0
                    hist_len = sum(steps_len[jidx]) if steps_len[jidx] else 0
                    mean_logp = hist_logp / (hist_len + 1e-8)
                    weights.append((np.exp(mean_logp) ** 0.5) * (tem_judge_p_list[jidx] ** 0.5))
                norm_w = softmax(np.array(weights))
                best_beam = int(np.random.choice(len(norm_w), p=norm_w))
                final_response = beam_responses[best_beam]
                cot = beam_trajs[best_beam]
                _t_sel += time.perf_counter() - _ts1
                print(f'  final judge weights={[round(w,3) for w in weights]} | best_beam={best_beam}',
                      flush=True)
                break
              except Exception as e:
                import traceback
                traceback.print_exc()
                try_count += 1
                print(f'  [ERROR attempt {try_count}/3] {e}', flush=True)
                torch.cuda.empty_cache()
                if try_count == 3:
                    final_response = ''
                    cot = ''

            elapsed = time.time() - t0
            torch.cuda.empty_cache()
            result = make_result(item, final_response, args.datasets,
                                 'base_gd', time_sec=elapsed, cot=cot)
            result['generation_time_seconds'] = round(_t_gen, 4)
            result['selection_time_seconds'] = round(_t_sel, 4)
            result['nemotron'] = {
                'method': 'gd', 'step_beam_size': args.step_beam_size,
                'num_rollout': args.num_rollout,
                'num_foresight': args.num_foresight,
                'runtime': 'transformers-trust-remote-code'}
            save_result(result, output_path)
            print(f"  response[:100]: {final_response[:100]!r} | {elapsed:.1f}s",
                  flush=True)

    with open(os.path.join(args.time_path, args.file_name + '.txt'), 'w') as f:
        f.write(f'total_time: {time.time()-start:.2f}s\n')
        f.write(f'step_beam_size: {args.step_beam_size}\n')
        f.write(f'num_rollout: {args.num_rollout}\n')
        f.write(f'num_foresight: {args.num_foresight}\n')
        f.write(f'total_input_tokens: {all_in}\n')
        f.write(f'total_output_tokens: {all_out}\n')
    print(f'\n[done] {time.time()-start:.1f}s | {output_path}', flush=True)


if __name__ == '__main__':
    main()
