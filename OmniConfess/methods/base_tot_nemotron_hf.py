# -*- coding: utf-8 -*-
"""
base_tot_nemotron_hf.py -- Tree of Thoughts baseline (step expansion + model
vote selection) on Nemotron-3-Nano-Omni-30B.

Algorithm 1:1 from base_tot_qwen3.py (step_beam_size=1, num_rollout=4,
num_foresight=3, n_vote=2, NOT retuned): per foresight step, sample
num_rollout candidate steps, model-vote the candidates n_vote times, keep the
top beam; final greedy completion + terminal vote.

Nemotron deltas vs the Qwen3 arm:
  * All generation goes through the outer omni wrapper's generate() (same path
    as base_sc_nemotron_hf.py) -- media is fused by the wrapper itself.
  * Rollouts are drawn SERIALLY (num_return_sequences batching is not
    exercised on this wrapper); each rollout is an independent do_sample call,
    semantically identical.
  * stop_strings are not passed (remote-code generate does not support the
    tokenizer/stop_strings kwargs reliably); the '\n\n' early stop is applied
    post-hoc by truncating the decoded step text at the first blank line.
  * No trajectory recording (Nemotron arm policy); gen/selection split times
    are recorded as in the Qwen3 arm.
"""
import os, sys, time, json, argparse, re, random, gc
import numpy as np
import torch

REPO = os.environ.get('NEMO_REPO', '/home/tsq/rhq/omniconfess/OmniConfess')
sys.path.insert(0, os.path.join(REPO, 'OmniConfess', 'methods'))
from shared import load_dataset, get_system_prompt
from shared.nemotron_shim import load_nemotron
from shared.nemotron_methods import (
    load_item_media, inputs_from_text, outer_generate, set_seed,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_VOTE_TEMPLATE = (
    "Given a problem and several choices of partial solution, decide which choice "
    "is most promising based on the video/audio content. Analyze each choice in "
    "detail, then conclude in the last line \"The best choice is {s}\", where s "
    "the integer id of the choice.\n"
)


def vote_prompt_wrap(question, candidates):
    p = _VOTE_TEMPLATE + f'\nProblem:\n{question}\n'
    for i, c in enumerate(candidates, 1):
        p += f'Choice {i}:\n{c}\n'
    return p


def parse_vote_output(text, n_candidates):
    m = re.search(r'best choice is\s*(\d+)', text, re.IGNORECASE | re.DOTALL)
    if m:
        v = int(m.group(1)) - 1
        if 0 <= v < n_candidates:
            return v
    return random.randrange(n_candidates)


def tally_votes(vote_outputs, n_candidates):
    counts = [0] * n_candidates
    for vout in vote_outputs:
        counts[parse_vote_output(vout, n_candidates)] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',       type=str, required=True)
    parser.add_argument('--data_path',      type=str, required=True)
    parser.add_argument('--data_root',      type=str, default=None)
    parser.add_argument('--output_dir',     type=str, default='./outputs')
    parser.add_argument('--file_name',      type=str, default='tot_nemotron')
    parser.add_argument('--time_path',      type=str, default='./outputs/time')
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--step_beam_size', type=int, default=1)
    parser.add_argument('--num_rollout',    type=int, default=4)
    parser.add_argument('--num_foresight',  type=int, default=3)
    parser.add_argument('--n_vote',         type=int, default=2)
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
    print(f'[tot] beam={args.step_beam_size} rollout={args.num_rollout} '
          f'foresight={args.num_foresight} n_vote={args.n_vote}', flush=True)

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
             temperature=0.7, stop_on_newline=False):
        nonlocal all_in, all_out
        inputs, il = inputs_from_text(processor, tokenizer, system_prompt,
                                      user_text, media, DEVICE,
                                      enable_thinking=args.enable_thinking)
        all_in += il
        text, gen_ids, _ = outer_generate(
            model, tokenizer, inputs, max_new_tokens=max_new_tokens,
            do_sample=do_sample, temperature=temperature)
        if stop_on_newline:
            text = text.split('\n\n')[0].strip()
        all_out += len(gen_ids)
        torch.cuda.empty_cache()
        return text

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
            final_response = ''
            cot = ''
            _t_gen = _t_sel = 0.0
            try:
                for T in range(args.num_foresight):
                    _tg0 = time.perf_counter()
                    all_candidates, all_chains = [], []
                    for beam_idx in range(args.step_beam_size):
                        step_text = (
                            f"Question: {q_text}\n"
                            f"{previous_steps[beam_idx]}\n"
                            "Please generate the NEXT logical reasoning step. Do NOT provide the final answer yet. Focus on analyzing the physical details step by step:")
                        for j in range(args.num_rollout):
                            resp = _gen(step_text, media, max_new_tokens=128,
                                        do_sample=True, temperature=0.7,
                                        stop_on_newline=True)
                            all_candidates.append(resp)
                            all_chains.append(previous_steps[beam_idx] + resp.strip() + '\n')
                    _t_gen += time.perf_counter() - _tg0

                    _ts0 = time.perf_counter()
                    vote_text = vote_prompt_wrap(q_text, all_chains)
                    vote_outputs = [_gen(vote_text, media, max_new_tokens=512,
                                         do_sample=True, temperature=0.7)
                                    for _ in range(args.n_vote)]
                    vote_counts = tally_votes(vote_outputs, len(all_candidates))
                    top_idx = np.array(vote_counts).argsort()[-args.step_beam_size:][::-1].tolist()
                    print(f'    votes={vote_counts} | selected={top_idx}', flush=True)
                    _t_sel += time.perf_counter() - _ts0
                    previous_steps = [all_chains[ti] for ti in top_idx]

                _tg1 = time.perf_counter()
                completed = []
                for beam_idx in range(len(previous_steps)):
                    final_text = (
                        f"Question: {q_text}\n"
                        f"{previous_steps[beam_idx]}\n"
                        "Please provide the final answer:")
                    resp = _gen(final_text, media, max_new_tokens=args.max_new_tokens,
                                do_sample=False)
                    completed.append(previous_steps[beam_idx] + '\n' + resp)
                _t_gen += time.perf_counter() - _tg1

                _ts1 = time.perf_counter()
                if len(completed) == 1:
                    final_response = completed[0].split('\n', completed[0].count('\n'))[-1].strip()
                    cot = completed[0]
                else:
                    tv_out = _gen(vote_prompt_wrap(q_text, completed), media,
                                  max_new_tokens=512, do_sample=False)
                    best = parse_vote_output(tv_out, len(completed))
                    cot = completed[best]
                    parts = completed[best].rsplit('\n', 1)
                    final_response = parts[-1].strip() if len(parts) > 1 else completed[best].strip()
                _t_sel += time.perf_counter() - _ts1
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f'  [ERROR] {e}', flush=True)
                final_response = ''
                cot = ''
            elapsed = time.time() - t0
            torch.cuda.empty_cache()

            from shared import make_result, save_result
            result = make_result(item, final_response, args.datasets,
                                 'base_tot', time_sec=elapsed, cot=cot)
            result['generation_time_seconds'] = round(_t_gen, 4)
            result['selection_time_seconds'] = round(_t_sel, 4)
            result['nemotron'] = {
                'method': 'tot',
                'step_beam_size': args.step_beam_size,
                'num_rollout': args.num_rollout,
                'num_foresight': args.num_foresight,
                'n_vote': args.n_vote,
                'runtime': 'transformers-trust-remote-code'}
            save_result(result, output_path)
            print(f"  response[:100]: {final_response[:100]!r} | {elapsed:.1f}s",
                  flush=True)

    with open(os.path.join(args.time_path, args.file_name + '.txt'), 'w') as f:
        f.write(f'total_time: {time.time()-start:.2f}s\n')
        f.write(f'step_beam_size: {args.step_beam_size}\n')
        f.write(f'num_rollout: {args.num_rollout}\n')
        f.write(f'num_foresight: {args.num_foresight}\n')
        f.write(f'n_vote: {args.n_vote}\n')
        f.write(f'total_input_tokens: {all_in}\n')
        f.write(f'total_output_tokens: {all_out}\n')
    print(f'\n[done] {time.time()-start:.1f}s | {output_path}', flush=True)


if __name__ == '__main__':
    main()
