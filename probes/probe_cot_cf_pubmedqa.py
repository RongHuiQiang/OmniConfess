# -*- coding: utf-8 -*-
"""probe_cot_cf_pubmedqa.py — per-token counterfactual (CF) probe over the
pubmedqa CoT trajectory, to give the problem-structure **sparsity axis** a
NATIVE 6th dataset (a judgment task with a real multi-token CoT to be sparse
over), instead of the -logp CoT-proxy used in leg2.

Design (reuses probe_claim_cf.py VERBATIM for the model + teacher-forcing, and
leg2_sparsity_math.py VERBATIM for the sparsity statistics — no new mechanism):

  - Trajectory: the stored greedy CoT `token_ids` from
    outputs/pubmedqa_base_model/pubmedqa_base_model_full_single_scored.jsonl
    (468/468 have token_ids; median 26 output tokens, 467 have >=10).
  - Views (same passage-swap contrast as probe_claim_cf.py's ragtruth path):
        full : own passage
        ctrl : a DIFFERENT item's passage (random, != own)
    delta_tok[i] = logP(tok_i | full) - logP(tok_i | ctrl)
    i.e. how much each CoT token's likelihood depends on the true evidence
    passage — the per-token counterfactual truth-move, exactly lens A.
  - Join: scored<->data.json by QUESTION TEXT (468/468; the integer `id` in the
    scored file and the `uid` in data.json do NOT correspond, and order differs
    — verified on disk before writing this).
  - Sparsity: |delta_tok| per sample -> participation_frac / hoyer / gini /
    top-k mass, pooled and per-sample, via leg2's formulas.

Zero new model code; single card; teacher-forced (no sampling).

    CUDA_VISIBLE_DEVICES=2 python probe_cot_cf_pubmedqa.py \
        --n_samples 200 --out outputs/problem_structure_fig/pubmedqa_cot_cf_tokens.jsonl
"""
import os, sys, json, argparse, random, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Reuse the probe_claim_cf machinery verbatim (model load, teacher-forced logp,
# view builder, prompt getters). Importing the module runs its sys.path setup.
import probe_claim_cf as P  # noqa: E402
from probe_claim_cf import (  # noqa: E402
    view_logp, build_view, load_model, get_system_prompt, get_user_prompt,
    token_class,
)

# Reuse leg2 sparsity formulas verbatim.
import leg2_sparsity_math as L2  # noqa: E402

import torch  # noqa: E402

# numpy 2.x removed np.trapz (renamed to np.trapezoid); leg2_sparsity_math.py
# still calls np.trapz. Shim it so we reuse leg2's formulas without editing it.
if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
    np.trapz = np.trapezoid


def load_pubmedqa():
    scored_path = os.path.join(
        HERE, 'outputs/pubmedqa_base_model/pubmedqa_base_model_full_single_scored.jsonl')
    data_path = os.path.join(HERE, 'OmniHalluBench/pubmedqa/annotations/pubmedqa.json')
    scored = [json.loads(l) for l in open(scored_path)]
    scored = [r for r in scored if r.get('token_ids')]
    data = json.load(open(data_path))
    by_q = {(d.get('question') or '').strip(): d for d in data}
    recs = []
    for r in scored:
        it = by_q.get((r.get('question') or '').strip())
        if it is None or not (it.get('passage') or ''):
            continue
        recs.append((r, it))
    return recs


def run(args):
    recs = load_pubmedqa()
    print(f'[cot-cf] pubmedqa joined+passaged records: {len(recs)}', flush=True)

    rng = random.Random(args.seed)
    # stratify by CoT length terciles so we sample short/mid/long CoT evenly
    recs_sorted = sorted(recs, key=lambda ri: len(ri[0]['token_ids']))
    k = max(1, len(recs_sorted) // 3)
    tiers = [recs_sorted[:k], recs_sorted[k:2 * k], recs_sorted[2 * k:]]
    for t in tiers:
        rng.shuffle(t)
    sample = []
    while len(sample) < args.n_samples:
        adv = False
        for t in tiers:
            if t and len(sample) < args.n_samples:
                sample.append(t.pop()); adv = True
        if not adv:
            break
    if args.limit:
        sample = sample[:args.limit]

    pool_passages = [it['passage'] for _, it in recs if it.get('passage')]

    processor, model = load_model()
    tokenizer = processor.tokenizer
    thinker = model.thinker
    system_prompt = get_system_prompt('pubmedqa', 'v1')

    out_f = open(args.out, 'w')
    t0 = time.time()
    n_done = 0
    tf_worst = 0.0
    for si, (rec, item) in enumerate(sample):
        gen_ids = torch.tensor(rec['token_ids'], dtype=torch.long, device=model.device)
        toks = [tokenizer.decode([t], skip_special_tokens=True) for t in rec['token_ids']]
        cls = [token_class(t, toks[i - 1] if i else None) for i, t in enumerate(toks)]

        # partner passage != own
        own = item.get('passage') or ''
        partner = None
        while partner is None:
            cand = rng.choice(pool_passages)
            if cand != own:
                partner = cand

        views = {}
        for name, psg in (('full', own), ('ctrl', partner)):
            up = get_user_prompt(item['question'], 'pubmedqa', 'v1', passage=psg)
            views[name] = build_view(processor, model, system_prompt, up, None, 'pubmedqa')

        lp = {n: view_logp(thinker, v, gen_ids) for n, v in views.items()}
        delta = (lp['full'] - lp['ctrl'])

        # teacher-force sanity: full-view logp must match the recorded logp_list
        if rec.get('logp_list') and len(rec['logp_list']) == len(lp['full']):
            tf = float(np.max(np.abs(lp['full'] - np.array(rec['logp_list']))))
            tf_worst = max(tf_worst, tf)

        out_f.write(json.dumps({
            'sample_i': si,
            'question': item['question'],
            'n_tok': len(rec['token_ids']),
            'delta_tok': [float(x) for x in delta],
            'classes': cls,
            'full_logp': [float(x) for x in lp['full']],
        }) + '\n')
        out_f.flush()
        n_done += 1
        if (si + 1) % 10 == 0:
            el = time.time() - t0
            print(f'[cot-cf] {si+1}/{len(sample)}  {el/ (si+1):.1f}s/samp  tf_worst={tf_worst:.4f}', flush=True)
    out_f.close()

    # ---- sparsity statistics (leg2 formulas, lens A = |delta_tok|) ----------
    samples = [json.loads(l) for l in open(args.out)]
    # pooled |delta| over content tokens (drop subword-continuation & punct so we
    # match leg2's "attribute/content token" footing; keep all if too few)
    def sample_vec(s):
        d = np.abs(np.asarray(s['delta_tok'], float))
        cl = s['classes']
        keep = np.array([c in ('content', 'num') for c in cl]) if len(cl) == len(d) else np.ones(len(d), bool)
        v = d[keep] if keep.sum() >= 5 else d
        return v
    per_sample = []
    pooled = []
    for s in samples:
        v = sample_vec(s)
        if v.size >= 5:
            per_sample.append(L2.participation_ratio(v) / v.size)
            pooled.append(v)
    pooled_vec = np.concatenate(pooled) if pooled else np.array([])
    stats = L2.all_stats(pooled_vec) if pooled_vec.size else {}
    ctrl = L2.random_control(pooled_vec) if pooled_vec.size else {}
    summary = {
        'dataset': 'pubmedqa',
        'lens': 'A (per-token CF |delta_tok|, passage-swap contrast)',
        'n_samples_used': len(per_sample),
        'n_tokens_pooled': int(pooled_vec.size),
        'tf_worst': tf_worst,
        'pooled_stats': stats,
        'random_control': ctrl,
        'per_sample_partic_frac_mean': float(np.mean(per_sample)) if per_sample else float('nan'),
    }
    js = args.out.replace('.jsonl', '_summary.json')
    json.dump(summary, open(js, 'w'), indent=2)
    print('[cot-cf] SUMMARY', json.dumps(summary, indent=2), flush=True)
    print('[cot-cf] wrote', args.out, 'and', js, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_samples', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(
        HERE, 'outputs/problem_structure_fig/pubmedqa_cot_cf_tokens.jsonl'))
    run(ap.parse_args())
