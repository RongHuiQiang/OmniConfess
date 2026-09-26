# -*- coding: utf-8 -*-
"""
base_pd_nemotron_hf.py -- Phi-Decoding baseline (multi-rollout + sigma-pruning
+ foresight completion + TF-IDF/KMeans clustering) on Nemotron-3-Nano-Omni-30B.

Algorithm 1:1 from base_pd_qwen3.py (step_beam_size=1, num_rollout=4,
num_foresight=3, cluster_num=2, sigma_rate=1.0, NOT retuned): per step, sample
rollout paths, sigma-prune by avg logp, complete each survivor, advantage =
completion avg logp - parent q-value, TF-IDF+KMeans cluster + advantage
softmax resample, final greedy answer.

Nemotron deltas vs the Qwen3 arm:
  * The Qwen3 arm batches prompts through the processor with padding and
    num_return_sequences; this wrapper's processor/generate path is exercised
    serially (one prompt, one rollout at a time). Each rollout/completion
    keeps its avg per-token logp from generate(output_scores=True) -- the same
    path score as compute_transition_scores(..., normalize_logits=True).
  * The assistant-prefix chat format (previous steps as a partial assistant
    turn) is applied via tokenizer.apply_chat_template with a plain-text
    assistant message, mirroring the Qwen3 arm's third message.
  * No trajectory recording (Nemotron arm policy).
"""
import os, sys, time, json, argparse, random
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

REPO = os.environ.get('NEMO_REPO', '/home/tsq/rhq/omniconfess/OmniConfess')
sys.path.insert(0, os.path.join(REPO, 'OmniConfess', 'methods'))
from shared import load_dataset, get_system_prompt, make_result, save_result
from shared.nemotron_shim import load_nemotron
from shared.nemotron_methods import (
    load_item_media, outer_generate, set_seed,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def softmax(x, temp=0.1):
    x = np.array(x, dtype=float)
    x = x / temp
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',        type=str, required=True)
    parser.add_argument('--data_path',       type=str, required=True)
    parser.add_argument('--data_root',       type=str, default=None)
    parser.add_argument('--output_dir',      type=str, default='./outputs')
    parser.add_argument('--file_name',       type=str, default='pd_nemotron')
    parser.add_argument('--time_path',       type=str, default='./outputs/time')
    parser.add_argument('--seed',            type=int, default=42)
    parser.add_argument('--step_beam_size',  type=int, default=1)
    parser.add_argument('--num_rollout',     type=int, default=4)
    parser.add_argument('--num_foresight',   type=int, default=3)
    parser.add_argument('--cluster_num',     type=int, default=2)
    parser.add_argument('--sigma_rate',      type=float, default=1.0)
    parser.add_argument('--max_new_tokens',  type=int, default=512)
    parser.add_argument('--prompt_version',  type=str, default='v1')
    parser.add_argument('--model_path',      type=str,
                        default=os.environ.get('NEMOTRON_MODEL_PATH'))
    parser.add_argument('--enable_thinking', action='store_true')
    parser.add_argument('--record_process',  action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--max_samples',     type=int, default=-1)
    parser.add_argument('--start_index',     type=int, default=0)
    parser.add_argument('--dry_run',         action='store_true')
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
    print(f'[pd] beam={args.step_beam_size} rollout={args.num_rollout} '
          f'foresight={args.num_foresight} cluster={args.cluster_num} '
          f'sigma={args.sigma_rate}', flush=True)

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

    def _inputs(user_text, media, assistant_prefix=None):
        """Processor inputs with media prefixes + optional partial assistant turn."""
        videos, images, audio = media
        prefix = ''
        if videos is not None: prefix += '<video>\n'
        if audio is not None:  prefix += '<so_embedding>\n'
        if images is not None: prefix += '<image>\n'
        user_full = prefix + user_text
        messages = [{'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_full}]
        if assistant_prefix is not None:
            messages.append({'role': 'assistant', 'content': assistant_prefix})
        try:
            text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=args.enable_thinking)
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
        kw = dict(text=[text], return_tensors='pt')
        if images is not None: kw['images'] = images
        if videos is not None: kw['videos'] = videos
        if audio is not None:  kw['audio'] = audio
        inputs = processor(**kw)
        inputs = {k: (v.to(DEVICE) if torch.is_tensor(v) else v)
                  for k, v in inputs.items()}
        for _mk in ('num_patches', 'num_tokens', 'imgs_sizes'):
            inputs.pop(_mk, None)
        return inputs, int(inputs['input_ids'].shape[1])

    def _gen(user_text, media, max_new_tokens=128, do_sample=True,
             temperature=0.6, assistant_prefix=None):
        nonlocal all_in, all_out
        inputs, il = _inputs(user_text, media, assistant_prefix)
        all_in += il
        text, gen_ids, avg_lp = outer_generate(
            model, tokenizer, inputs, max_new_tokens=max_new_tokens,
            do_sample=do_sample, temperature=temperature, want_logp=True)
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
            previous_q_vals = [0.0] * args.step_beam_size
            final_response = ''
            cot = ''
            _t_gen = _t_sel = 0.0
            try_count = 0

            while try_count < 3:
              try:
                for T in range(args.num_foresight):
                    print(f'  foresight {T+1}/{args.num_foresight}', flush=True)
                    # 1+2. rollouts per beam (serial), with path avg logp
                    _tg0 = time.perf_counter()
                    cur_logps, cur_texts = [], []
                    for beam_idx in range(args.step_beam_size):
                        step_prompt = (
                            f"Problem: {q_text}\n"
                            f"{previous_steps[beam_idx]}\n"
                            "Please generate the NEXT logical reasoning step. Do NOT provide the final answer yet. Focus strictly on analyzing the physical details step by step.")
                        for j in range(args.num_rollout):
                            resp, avg_lp = _gen(step_prompt, media,
                                                max_new_tokens=128,
                                                do_sample=True, temperature=0.6)
                            resp = resp.split('\n\n')[0].strip()
                            cur_texts.append(resp)
                            cur_logps.append(float(avg_lp) if avg_lp is not None else 0.0)
                    _t_gen += time.perf_counter() - _tg0
                    n_total = len(cur_texts)

                    # 3. sigma pruning
                    _ts0 = time.perf_counter()
                    arr = np.array(cur_logps)
                    mean, std = np.mean(arr), np.std(arr)
                    keep = [i for i, v in enumerate(cur_logps)
                            if v > mean - args.sigma_rate * std]
                    if len(keep) < args.step_beam_size:
                        keep = list(np.argsort(cur_logps)[-max(args.step_beam_size, 1):])
                    print(f'    sigma-prune: {len(keep)}/{n_total} survived', flush=True)

                    # 4. foresight completion of each survivor
                    _t_sel += time.perf_counter() - _ts0   # sigma-pruning time
                    _tg1 = time.perf_counter()
                    comp_user = (f"Problem: {q_text}\n"
                                 "Please directly output the remaining reasoning steps.")
                    comp_logps, comp_texts = [], []
                    for ki in keep:
                        parent_beam = ki // args.num_rollout
                        chain = previous_steps[parent_beam] + cur_texts[ki] + '\n'
                        fr, fa = _gen(comp_user, media, max_new_tokens=512,
                                      do_sample=True, temperature=0.7,
                                      assistant_prefix=chain)
                        comp_texts.append(fr)
                        comp_logps.append(float(fa) if fa is not None else 0.0)
                    _t_gen += time.perf_counter() - _tg1

                    # 5. cluster + advantage-weighted selection
                    _ts1 = time.perf_counter()
                    advantages = [comp_logps[i] - previous_q_vals[ki // args.num_rollout]
                                  for i, ki in enumerate(keep)]
                    valid = [(i, comp_texts[i], advantages[i])
                             for i in range(len(comp_texts))
                             if comp_texts[i].strip() and advantages[i] > 0]
                    if len(valid) < args.step_beam_size:
                        valid = [(i, comp_texts[i], advantages[i])
                                 for i in range(len(comp_texts)) if comp_texts[i].strip()]

                    selected_comp = list(range(min(args.step_beam_size, len(keep))))
                    if len(valid) >= args.step_beam_size:
                        v_indices = [v[0] for v in valid]
                        v_texts = [v[1] for v in valid]
                        v_advs = [v[2] for v in valid]
                        try:
                            k_val = min(args.cluster_num, len(v_texts))
                            vect = TfidfVectorizer(min_df=1)
                            X = vect.fit_transform(v_texts)
                            km = KMeans(n_clusters=k_val, n_init='auto',
                                        random_state=args.seed)
                            km.fit(X)
                            labels = km.labels_
                            counts = np.bincount(labels, minlength=k_val)
                            cluster_w = softmax([counts[l] for l in labels])
                            adv_w = softmax(v_advs)
                            final_w = (cluster_w + adv_w) / 2
                            sel = np.random.choice(
                                len(final_w), p=final_w,
                                size=args.step_beam_size, replace=False).tolist()
                            selected_comp = [v_indices[s] for s in sel]
                        except Exception as e:
                            print(f'    cluster failed: {e}, adv fallback', flush=True)
                            adv_w = softmax(v_advs)
                            sel = np.random.choice(len(adv_w), p=adv_w,
                                                   size=args.step_beam_size,
                                                   replace=False).tolist()
                            selected_comp = [v_indices[s] for s in sel]

                    new_steps, new_qvals = [], []
                    for sci in selected_comp:
                        ki = keep[sci]
                        parent_beam = ki // args.num_rollout
                        new_steps.append(previous_steps[parent_beam] + cur_texts[ki] + '\n')
                        new_qvals.append(comp_logps[sci])
                    previous_steps, previous_q_vals = new_steps, new_qvals
                    _t_sel += time.perf_counter() - _ts1

                # final greedy answer
                _tg2 = time.perf_counter()
                final_step_prompt = (
                    f"Problem: {q_text}\n"
                    f"{previous_steps[0]}\n"
                    "Please provide the final answer:")
                final_response, _ = _gen(final_step_prompt, media,
                                         max_new_tokens=args.max_new_tokens,
                                         do_sample=False)
                cot = previous_steps[0] + '\n' + final_response
                _t_gen += time.perf_counter() - _tg2
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
                                 'base_pd', time_sec=elapsed, cot=cot)
            result['generation_time_seconds'] = round(_t_gen, 4)
            result['selection_time_seconds'] = round(_t_sel, 4)
            result['nemotron'] = {
                'method': 'pd', 'step_beam_size': args.step_beam_size,
                'num_rollout': args.num_rollout,
                'num_foresight': args.num_foresight,
                'cluster_num': args.cluster_num,
                'sigma_rate': args.sigma_rate,
                'runtime': 'transformers-trust-remote-code'}
            save_result(result, output_path)
            print(f"  response[:100]: {final_response[:100]!r} | {elapsed:.1f}s",
                  flush=True)

    with open(os.path.join(args.time_path, args.file_name + '.txt'), 'w') as f:
        f.write(f'total_time: {time.time()-start:.2f}s\n')
        f.write(f'step_beam_size: {args.step_beam_size}\n')
        f.write(f'num_rollout: {args.num_rollout}\n')
        f.write(f'num_foresight: {args.num_foresight}\n')
        f.write(f'cluster_num: {args.cluster_num}\n')
        f.write(f'sigma_rate: {args.sigma_rate}\n')
        f.write(f'total_input_tokens: {all_in}\n')
        f.write(f'total_output_tokens: {all_out}\n')
    print(f'\n[done] {time.time()-start:.1f}s | {output_path}', flush=True)


if __name__ == '__main__':
    main()
