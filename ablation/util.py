#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared helpers for the ABLATION_PLAN.md experiments (A1-A10).

Conventions:
  * probe phd delta_noctx = lp_yes_full - lp_yes_noctx  (probe convention)
    operator Delta_cf     = lp_yes(cf view) - lp_yes(full) = -delta_noctx
  * cmm probe d_cf_* are already operator-convention (masked - full),
    d_cf_sum = d_cf_audio + d_cf_video
  * pairing for judgement full files: identical row order (verified by
    identical (question, gt) sequences); for qwen3 GPFS files same check.
"""
import json
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def dump_jsonl(path, rows):
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def bootstrap_auc(x, y, n=2000, seed=0):
    """AUC with percentile bootstrap CI."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    rng = np.random.RandomState(seed)
    aucs = []
    m = len(x)
    for _ in range(n):
        idx = rng.randint(0, m, m)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        aucs.append(roc_auc_score(yy, x[idx]))
    pt = roc_auc_score(y, x)
    return pt, (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def mcnemar_exact(b, c):
    """Exact binomial McNemar p for discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    return stats.binomtest(min(b, c), n, 0.5).pvalue


def judgement_preds(rows):
    """(gt, pred) binary lists from *_single_scored rows, maybe-gt excluded,
    maybe-pred counted as a miss (identical to single_eval.py)."""
    out = []
    for r in rows:
        em = r.get('eval_metrics')
        if em is None:
            continue
        g, p = em['Ground_Truth'], em['Prediction']
        if g == 2:
            continue
        p = (1 - g) if p == 2 else p
        out.append((g, p))
    return out


def f1_yp(gts, preds):
    from sklearn.metrics import f1_score
    return f1_score(gts, preds, zero_division=0) * 100


def paired_mcnemar_from_preds(preds_a, preds_b, gts):
    """b = a wrong & b right, c = a right & b wrong (a vs b)."""
    b = sum(1 for g, pa, pb in zip(gts, preds_a, preds_b) if pa != g and pb == g)
    c = sum(1 for g, pa, pb in zip(gts, preds_a, preds_b) if pa == g and pb != g)
    return b, c, mcnemar_exact(b, c)


def align_rows(*files, key=('question', 'gt')):
    """Verify all files share the same row order under key; return rows lists."""
    loaded = [load_jsonl(f) for f in files]
    ref = [tuple(str(r.get(k)) for k in key) for r in loaded[0]]
    for f, rows in zip(files[1:], loaded[1:]):
        k2 = [tuple(str(r.get(k)) for k in key) for r in rows]
        if k2 != ref:
            raise ValueError(f'row order mismatch: {f}')
    return loaded


def pair_by_question(*files, drop_dups=False):
    """Pair rows of multiple files to the FIRST file's row order, matching by
    (question, gt) with positional occurrence index. Duplicated (q, gt) keys
    pair in file order. Verifies gt agreement on aligned pairs."""
    from collections import defaultdict

    def key(r):
        return (r['question'], str(r['gt']).strip().lower())
    loaded = [load_jsonl(f) for f in files]
    first = loaded[0]
    out = [first]
    for rows in loaded[1:]:
        g = defaultdict(list)
        for r in rows:
            g[key(r)].append(r)
        paired = []
        for r in first:
            grp = g[key(r)]
            if not grp:
                raise ValueError('cannot pair: (question,gt) multiset differs')
            paired.append(grp.pop(0))
        for r0, r1 in zip(first, paired):
            if str(r0['gt']).strip().lower() != str(r1['gt']).strip().lower():
                raise ValueError('gt mismatch on paired question')
        out.append(paired)
    return out


def token_f1_per_row(path):
    """Token-F1 per row from a descriptive *_single_scored file."""
    rows = load_jsonl(path)
    return {r['question']: r['eval_metrics']['Token-F1'] for r in rows
            if 'eval_metrics' in r}


def paired_perm_test(d, n=10000, seed=0):
    """One-sided sign-flip permutation test for mean(d) > 0."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float('nan'), float('nan')
    rng = np.random.RandomState(seed)
    obs = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = (signs * d).mean(axis=1)
    p = float((null >= obs).mean())
    return obs, p
