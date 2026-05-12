# -*- coding: utf-8 -*-
"""
base_pd.py — Phi-Decoding: multi-rollout + σ-pruning + TF-IDF/KMeans clustering.

Algorithm per foresight step T:
  1. Generate step_beam_size * num_rollout candidate steps (batched)
  2. σ-pruning: keep paths with avg_logp > mean - sigma_rate * std
  3. Foresight completion: complete each surviving path
  4. Advantage = completion_avg_logp - parent_q_value
  5. TF-IDF + KMeans cluster (cluster_num clusters)
     -> weight = 0.5 * cluster_ratio + 0.5 * adv_softmax
     -> resample step_beam_size paths
  Final: generate answer from best beam (greedy)

Usage:
    python base_pd.py --datasets cmm --data_path /path/to/cmm.json --dry_run
"""
import os, sys, time, argparse, random, json
import torch, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

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


def softmax(x, temp=0.1):
    x = np.array(x, dtype=float)
    x = x / temp
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',        type=str, required=True)
    parser.add_argument('--data_path',       type=str, required=True)
    parser.add_argument('--data_root',       type=str, default='./OmniHalluBench/')
    parser.add_argument('--output_dir',      type=str, default='./results')
    parser.add_argument('--file_name',       type=str, default='base_pd_out')
    parser.add_argument('--time_path',       type=str, default='./results/time')
    parser.add_argument('--seed',            type=int, default=42)
    parser.add_argument('--step_beam_size',  type=int, default=1)
    parser.add_argument('--num_rollout',     type=int, default=4)
    parser.add_argument('--num_foresight',   type=int, default=3)
    parser.add_argument('--cluster_num',     type=int, default=2)
    parser.add_argument('--sigma_rate',      type=float, default=1.0)
    parser.add_argument('--max_new_tokens',  type=int, default=512)
    parser.add_argument('--prompt_version',  type=str, default='v1', choices=['v0','v1','v2','v3'])
    parser.add_argument('--dry_run',         action='store_true')
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

        q_text   = item['question']
        passage  = item.get('passage')

        # Beam state
        previous_steps  = ['The reasoning steps are:\n\n'] * args.step_beam_size
        previous_q_vals = [0.0] * args.step_beam_size

        final_response = ''
        cot = ''
        try_count = 0
        _t_gen = 0.0
        _t_sel = 0.0

        while try_count < 3:
          try:
            for T in range(args.num_foresight):
                print(f'  foresight {T+1}/{args.num_foresight}', flush=True)

                # ── 1. Build batch prompts for all beams ─────────────────────
                prompts = []
                for beam_idx in range(args.step_beam_size):
                    step_prompt = (
                        f"Problem: {q_text}\n"
                        f"{previous_steps[beam_idx]}\n"
                        "Please generate the NEXT logical reasoning step. Do NOT provide the final answer yet. Focus strictly on analyzing the physical details step by step."
                    )
                    user_content = build_user_content_with_media_tokens(
                        pixel_values, audio_values, image_values, step_prompt)
                    chat = [
                        {'role': 'system',    'content': [{'type': 'text', 'text': system_prompt}]},
                        {'role': 'user',      'content': [{'type': 'text', 'text': user_content}]},
                        {'role': 'assistant', 'content': [{'type': 'text', 'text': previous_steps[beam_idx]}]},
                    ]
                    p_str = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                    if isinstance(p_str, list): p_str = p_str[0]
                    p_str = p_str.replace(tokenizer.eos_token, '').strip()
                    prompts.append(p_str)

                # Build batched inputs
                n = len(prompts)
                pk = {'text': prompts, 'return_tensors': 'pt',
                      'padding': True, 'fps': [1.0] * n}
                if pixel_values is not None: pk['videos'] = [pixel_values] * n
                if audio_values is not None: pk['audio']  = [audio_values] * n
                if image_values is not None: pk['images'] = [image_values] * n
                if 'halo' in args.datasets.lower(): pk['max_pixels'] = 224 * 224
                m_inputs = processor(**pk).to(model.device)
                clean_m = prepare_for_thinker(m_inputs, model)
                input_len = clean_m['input_ids'].shape[1]
                all_input_tokens += clean_m['input_ids'].numel()

                # ── 2. Generate num_rollout paths per beam ───────────────────
                _tg0 = time.perf_counter()
                with torch.no_grad():
                    rollout_out = model.thinker.generate(
                        **clean_m,
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=0.6,
                        num_return_sequences=args.num_rollout,
                        output_scores=True,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.eos_token_id,
                        stop_strings=['<end_of_reasoning>', '\n\n'],
                        tokenizer=tokenizer,
                    )
                    trans_scores = model.thinker.compute_transition_scores(
                        rollout_out.sequences, rollout_out.scores, normalize_logits=True)

                cur_logps, cur_lens, cur_texts = [], [], []
                n_total = args.step_beam_size * args.num_rollout
                for r_idx in range(n_total):
                    gen_ids = rollout_out.sequences[r_idx][input_len:]
                    valid_mask = (gen_ids != tokenizer.pad_token_id)
                    gen_ids = gen_ids[valid_mask]
                    resp = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    path_scores = trans_scores[r_idx][:valid_mask.sum()]
                    cum_logp = path_scores.sum().item()
                    tok_cnt = len(gen_ids)
                    cur_logps.append(cum_logp / (tok_cnt + 1e-8))
                    cur_lens.append(tok_cnt)
                    cur_texts.append(resp)
                    all_output_tokens += tok_cnt

                torch.cuda.empty_cache()
                _t_gen += time.perf_counter() - _tg0

                # ── 3. σ-pruning ─────────────────────────────────────────────
                _ts0 = time.perf_counter()
                arr = np.array(cur_logps)
                mean, std = np.mean(arr), np.std(arr)
                keep = [i for i, v in enumerate(cur_logps)
                        if v > mean - args.sigma_rate * std]
                if len(keep) < args.step_beam_size:
                    # Fallback: keep all sorted by logp
                    keep = list(np.argsort(cur_logps)[-max(args.step_beam_size, 1):])
                print(f'    σ-prune: {len(keep)}/{n_total} survived', flush=True)

                # ── 4. Foresight completion (complete each surviving path) ────
                comp_prompts = []
                for ki in keep:
                    parent_beam = ki // args.num_rollout
                    chain = previous_steps[parent_beam] + cur_texts[ki] + '\n'
                    comp_user = build_user_content_with_media_tokens(
                        pixel_values, audio_values, image_values,
                        f"Problem: {q_text}\nPlease directly output the remaining reasoning steps.")
                    comp_chat = [
                        {'role': 'system',    'content': [{'type': 'text', 'text': system_prompt}]},
                        {'role': 'user',      'content': [{'type': 'text', 'text': comp_user}]},
                        {'role': 'assistant', 'content': [{'type': 'text', 'text': chain}]},
                    ]
                    cp_str = processor.apply_chat_template(comp_chat, tokenize=False, add_generation_prompt=True)
                    if isinstance(cp_str, list): cp_str = cp_str[0]
                    comp_prompts.append(cp_str.replace(tokenizer.eos_token, '').strip())

                _t_sel += time.perf_counter() - _ts0  # σ-pruning time

                _tg1 = time.perf_counter()
                nc = len(comp_prompts)
                cpk = {'text': comp_prompts, 'return_tensors': 'pt',
                       'padding': True, 'truncation': True, 'max_length': 2048,
                       'fps': [1.0] * nc}
                if pixel_values is not None: cpk['videos'] = [pixel_values] * nc
                if audio_values is not None: cpk['audio']  = [audio_values] * nc
                if image_values is not None: cpk['images'] = [image_values] * nc
                if 'halo' in args.datasets.lower(): cpk['max_pixels'] = 224 * 224
                c_inputs = processor(**cpk).to(model.device)
                clean_c = prepare_for_thinker(c_inputs, model)

                with torch.no_grad():
                    f_out = model.thinker.generate(
                        **clean_c,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        output_scores=True,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    f_trans = model.thinker.compute_transition_scores(
                        f_out.sequences, f_out.scores, normalize_logits=True)

                f_input_len = clean_c['input_ids'].shape[1]
                comp_logps, comp_texts, advantages = [], [], []
                for jf, ki in enumerate(keep):
                    parent_beam = ki // args.num_rollout
                    fg = f_out.sequences[jf][f_input_len:]
                    fm = (fg != tokenizer.pad_token_id)
                    fg = fg[fm]
                    fr = tokenizer.decode(fg, skip_special_tokens=True).strip()
                    fp = (f_trans[jf][:fm.sum()]).sum().item()
                    fa = fp / (len(fg) + 1e-8)
                    comp_logps.append(fa)
                    comp_texts.append(fr)
                    advantages.append(fa - previous_q_vals[parent_beam])
                    all_output_tokens += len(fg)

                torch.cuda.empty_cache()
                _t_gen += time.perf_counter() - _tg1

                # ── 5. Cluster + advantage-weighted selection ────────────────
                _ts1 = time.perf_counter()
                valid = [(i, comp_texts[i], advantages[i])
                         for i in range(len(comp_texts))
                         if comp_texts[i].strip() and advantages[i] > 0]
                if len(valid) < args.step_beam_size:
                    valid = [(i, comp_texts[i], advantages[i])
                             for i in range(len(comp_texts)) if comp_texts[i].strip()]

                selected_keep_indices = list(range(min(args.step_beam_size, len(keep))))
                if len(valid) >= args.step_beam_size:
                    v_indices = [v[0] for v in valid]
                    v_texts   = [v[1] for v in valid]
                    v_advs    = [v[2] for v in valid]
                    try:
                        k_val = min(args.cluster_num, len(v_texts))
                        vect  = TfidfVectorizer(min_df=1)
                        X     = vect.fit_transform(v_texts)
                        km    = KMeans(n_clusters=k_val, n_init='auto', random_state=args.seed)
                        km.fit(X)
                        labels = km.labels_
                        counts = np.bincount(labels, minlength=k_val)
                        cluster_w = softmax([counts[l] for l in labels])
                        adv_w     = softmax(v_advs)
                        final_w   = (cluster_w + adv_w) / 2
                        sel = np.random.choice(
                            len(final_w), p=final_w,
                            size=args.step_beam_size, replace=False).tolist()
                        selected_keep_indices = [v_indices[s] for s in sel]
                    except Exception as e:
                        print(f'    cluster failed: {e}, using adv fallback', flush=True)
                        adv_w = softmax(v_advs)
                        sel = np.random.choice(len(adv_w), p=adv_w,
                                               size=args.step_beam_size, replace=False).tolist()
                        selected_keep_indices = [v_indices[s] for s in sel]

                # ── Update beams ────────────────────────────────────────────
                new_steps, new_qvals = [], []
                for ski in selected_keep_indices:
                    ki = keep[ski]
                    parent_beam = ki // args.num_rollout
                    new_steps.append(
                        previous_steps[parent_beam] + cur_texts[ki] + '\n')
                    new_qvals.append(comp_logps[ski])

                previous_steps  = new_steps
                previous_q_vals = new_qvals
                _t_sel += time.perf_counter() - _ts1

            # ── Final generation ─────────────────────────────────────────────
            _tg2 = time.perf_counter()
            final_step_prompt = (
                f"Problem: {q_text}\n"
                f"{previous_steps[0]}\n"
                "Please provide the final answer:"
            )
            final_content = build_user_content_with_media_tokens(
                pixel_values, audio_values, image_values, final_step_prompt)
            final_chat = [
                {'role': 'system',    'content': [{'type': 'text', 'text': system_prompt}]},
                {'role': 'user',      'content': [{'type': 'text', 'text': final_content}]},
            ]
            fp_str = processor.apply_chat_template(final_chat, tokenize=False, add_generation_prompt=True)
            if isinstance(fp_str, list): fp_str = fp_str[0]
            fp_str = fp_str.replace(tokenizer.eos_token, '').strip()

            fpk = {'text': [fp_str], 'return_tensors': 'pt', 'fps': [1.0]}
            if pixel_values is not None: fpk['videos'] = [pixel_values]
            if audio_values is not None: fpk['audio']  = [audio_values]
            if image_values is not None: fpk['images'] = [image_values]
            if 'halo' in args.datasets.lower(): fpk['max_pixels'] = 224 * 224
            fp_inputs = processor(**fpk).to(model.device)
            fp_clean = prepare_for_thinker(fp_inputs, model)
            fp_input_len = fp_clean['input_ids'].shape[1]

            with torch.inference_mode():
                fp_out = model.thinker.generate(
                    **fp_clean,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            fp_gen = fp_out[0][fp_input_len:]
            final_response = tokenizer.decode(fp_gen, skip_special_tokens=True).strip()
            all_output_tokens += len(fp_gen)
            cot = previous_steps[0] + '\n' + final_response
            _t_gen += time.perf_counter() - _tg2

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

        result = make_result(item, final_response, args.datasets, 'base_pd',
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
        f.write(f'cluster_num: {args.cluster_num}\n')
        f.write(f'sigma_rate: {args.sigma_rate}\n')
        f.write(f'total_input_tokens: {all_input_tokens}\n')
        f.write(f'total_output_tokens: {all_output_tokens}\n')
    print(f'\n[done] {total_time:.1f}s | {output_path}', flush=True)


if __name__ == '__main__':
    main()
