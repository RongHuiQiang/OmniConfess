# -*- coding: utf-8 -*-
"""
base_tot.py — Tree of Thoughts: step expansion + model vote selection.

Algorithm per foresight step T:
  1. For each beam, generate num_rollout candidate steps (serial sampling)
  2. Build vote prompt listing all candidates
  3. Model votes n_vote times -> parse "The best choice is {id}"
  4. Select top step_beam_size by vote count -> update beams
  Final: complete each surviving beam (greedy), terminal vote picks best answer.

Usage:
    python base_tot.py --datasets cmm --data_path /path/to/cmm.json --dry_run
"""
import os, sys, time, argparse, random, json, re
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

_VOTE_TEMPLATE = (
    "Given a problem and several choices of partial solution, decide which choice "
    "is most promising based on the video/audio content. Analyze each choice in "
    "detail, then conclude in the last line \"The best choice is {s}\", where s "
    "the integer id of the choice.\n"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def vote_prompt_wrap(question, candidates):
    p = _VOTE_TEMPLATE + f'\nProblem:\n{question}\n'
    for i, c in enumerate(candidates, 1):
        p += f'Choice {i}:\n{c}\n'
    return p


def parse_vote_output(text, n_candidates):
    """Return 0-indexed winner, or random fallback."""
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
    parser.add_argument('--data_root',      type=str, default='./OmniHalluBench/')
    parser.add_argument('--output_dir',     type=str, default='./results')
    parser.add_argument('--file_name',      type=str, default='base_tot_out')
    parser.add_argument('--time_path',      type=str, default='./results/time')
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--step_beam_size', type=int, default=1)
    parser.add_argument('--num_rollout',    type=int, default=4)
    parser.add_argument('--num_foresight',  type=int, default=3)
    parser.add_argument('--n_vote',         type=int, default=2,
                        help='Number of times model votes per step')
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

    def _encode(messages, pv, av, iv):
        """Build + prepare inputs for thinker.generate()."""
        raw = build_inputs(processor, model, messages, pv, av, iv, args.datasets)
        clean = prepare_for_thinker(raw, model)
        return clean

    def _gen(clean, max_new_tokens=128, do_sample=True, temperature=0.7,
             stop_on_newline=False):
        nonlocal all_input_tokens, all_output_tokens
        all_input_tokens += clean['input_ids'].shape[1]
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id,
            stop_strings=['<end_of_reasoning>'],
            tokenizer=tokenizer,
        )
        if stop_on_newline:
            gen_kwargs['stop_strings'] = ['\n\n', '<end_of_reasoning>']
        with torch.inference_mode():
            out = model.thinker.generate(**clean, **gen_kwargs)
        il = clean['input_ids'].shape[1]
        gen_ids = out[0][il:]
        gen_ids = gen_ids[gen_ids != tokenizer.pad_token_id]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        all_output_tokens += len(gen_ids)
        torch.cuda.empty_cache()
        return text

    for idx, item in enumerate(data):
        if item['id'] in done_ids:
            continue

        t0 = time.time()
        print(f'\n[{idx+1}/{len(data)}] id={item["id"]}', flush=True)

        pixel_values, audio_values, image_values = load_media_for_item(
            item, args.data_path, args.data_root)

        q_text  = item['question']
        passage = item.get('passage')

        # Beam state: step_beam_size parallel reasoning chains
        previous_steps = ['The reasoning steps are:\n\n'] * args.step_beam_size

        final_response = ''
        cot = ''
        _t_gen = 0.0
        _t_sel = 0.0

        try:
            for T in range(args.num_foresight):
                print(f'  foresight {T+1}/{args.num_foresight}', flush=True)

                # ── 1. Generate candidates for each beam ─────────────────────
                _tg0 = time.perf_counter()
                all_candidates = []          # flat list
                all_chains     = []          # chain = previous_steps[beam] + step
                parent_map     = []          # candidate_idx -> beam_idx

                for beam_idx in range(args.step_beam_size):
                    step_text = (
                        f"Question: {q_text}\n"
                        f"{previous_steps[beam_idx]}\n"
                        "Please generate the NEXT logical reasoning step. Do NOT provide the final answer yet. Focus on analyzing the physical details step by step:"
                    )
                    uc = build_user_content_with_media_tokens(
                        pixel_values, audio_values, image_values, step_text)
                    msgs = make_messages(system_prompt, uc)
                    clean = _encode(msgs, pixel_values, audio_values, image_values)

                    for j in range(args.num_rollout):
                        print(f'    beam {beam_idx} rollout {j+1}', flush=True)
                        resp = _gen(clean, max_new_tokens=128, do_sample=True, temperature=0.7,
                                    stop_on_newline=True)
                        all_candidates.append(resp)
                        all_chains.append(previous_steps[beam_idx] + resp.strip() + '\n')
                        parent_map.append(beam_idx)

                _t_gen += time.perf_counter() - _tg0

                # ── 2. Vote: model selects best candidate(s) ─────────────────
                _ts0 = time.perf_counter()
                vote_prompt_text = vote_prompt_wrap(q_text, all_chains)
                vote_uc = build_user_content_with_media_tokens(
                    pixel_values, audio_values, image_values, vote_prompt_text)
                vote_msgs = make_messages(system_prompt, vote_uc)
                vote_clean = _encode(vote_msgs, pixel_values, audio_values, image_values)

                vote_outputs = []
                for _ in range(args.n_vote):
                    vo = _gen(vote_clean, max_new_tokens=512, do_sample=True, temperature=0.7)
                    vote_outputs.append(vo)

                vote_counts = tally_votes(vote_outputs, len(all_candidates))
                arr = np.array(vote_counts)
                # Top step_beam_size by vote count
                top_idx = arr.argsort()[-args.step_beam_size:][::-1].tolist()
                print(f'    votes={vote_counts} | selected={top_idx}', flush=True)

                _t_sel += time.perf_counter() - _ts0

                # ── 3. Update beams ──────────────────────────────────────────
                new_steps = [all_chains[ti] for ti in top_idx]
                previous_steps = new_steps

            # ── Final completion ─────────────────────────────────────────────
            _tg1 = time.perf_counter()
            print(f'  final completion for {args.step_beam_size} beams', flush=True)
            completed = []
            for beam_idx in range(len(previous_steps)):
                final_text = (
                    f"Question: {q_text}\n"
                    f"{previous_steps[beam_idx]}\n"
                    "Please provide the final answer:"
                )
                fc = build_user_content_with_media_tokens(
                    pixel_values, audio_values, image_values, final_text)
                fm = make_messages(system_prompt, fc)
                fclean = _encode(fm, pixel_values, audio_values, image_values)
                resp = _gen(fclean, max_new_tokens=args.max_new_tokens, do_sample=False)
                completed.append(previous_steps[beam_idx] + '\n' + resp)

            _t_gen += time.perf_counter() - _tg1

            # ── Terminal vote (if multiple beams) ────────────────────────────
            _ts1 = time.perf_counter()
            if len(completed) == 1:
                final_response = completed[0].split('\n', completed[0].count('\n'))[-1].strip()
                cot = completed[0]
            else:
                tv_text = vote_prompt_wrap(q_text, completed)
                tv_uc   = build_user_content_with_media_tokens(
                    pixel_values, audio_values, image_values, tv_text)
                tv_msgs = make_messages(system_prompt, tv_uc)
                tv_clean = _encode(tv_msgs, pixel_values, audio_values, image_values)
                tv_out = _gen(tv_clean, max_new_tokens=512, do_sample=False)
                best = parse_vote_output(tv_out, len(completed))
                cot = completed[best]
                # Extract just the final answer (last part after reasoning chain)
                parts = completed[best].rsplit('\n', 1)
                final_response = parts[-1].strip() if len(parts) > 1 else completed[best].strip()
            _t_sel += time.perf_counter() - _ts1

        except Exception as e:
            print(f'  [ERROR] {e}', flush=True)
            final_response = ''
            cot = ''

        elapsed = time.time() - t0
        torch.cuda.empty_cache()
        print(f'  response[:100]: {final_response[:100]!r} | time={elapsed:.1f}s', flush=True)

        result = make_result(item, final_response, args.datasets, 'base_tot',
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
        f.write(f'n_vote: {args.n_vote}\n')
        f.write(f'total_input_tokens: {all_input_tokens}\n')
        f.write(f'total_output_tokens: {all_output_tokens}\n')
    print(f'\n[done] {total_time:.1f}s | {output_path}', flush=True)


if __name__ == '__main__':
    main()
