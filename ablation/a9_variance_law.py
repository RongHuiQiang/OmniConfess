#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A9 (ABLATION_PLAN.md): wording-vs-evidence variance law, continuous test.

Law (binary observation so far): BoN-style selection gains exist when the
candidate pool varies WORDING over the SAME evidence (temperature pools:
ragtruth +3.86), and dies when the pool varies EVIDENCE (avhbench time-window
pools: gate2 dead). Upgrade to a continuous relation:

  per-sample x = mean pairwise Jaccard(candidate_i, candidate_j) over the
                 pool's content words   (0 = disjoint evidence, 1 = same
                 evidence, different wording)
  per-sample y = selected-candidate Token-F1 - greedy-candidate Token-F1
                 (the selection gain the ranker actually banks)

Pools on disk (all zero-GPU):
  ragtruth v1 temp pool   (wording variance, expected positive slope)
  haloquest v1 temp pool  (mixed)
  haloquest v2 temp pool  (81 release-branch items, contract pool)
  avhbench window pool    (evidence variance, expected flat/negative)
  avhbench temp arms      (2 arms, wording variance within same input)

Prediction (pre-registered): slope of y on x > 0 for wording-type pools and
~0/<0 for the window pool; the relation pooled across pools is monotone.
"""
import os
import sys
import json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import load_jsonl

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(HERE))
sys.path.insert(0, os.getcwd())
from answerability_gate import token_f1  # noqa: E402

OUT = os.path.join(HERE, 'a9_variance_law.json')

STOP = set('''a an the and or of to in on at as is are was were be been being it its this that these those with while by for from into near nearby
there here his her their our your my he she they we you i who which what when where how not no nor but so then than also very just only
one two single describe see hear seen heard video audio image picture clip scene shows show can cannot could would should may might'''.split())
PUNCT = "!\"#$%&'()*+,-./:;<=>?@[]^_`{|}~\\"


def cwords(s):
    s = str(s).lower().translate(str.maketrans('', '', PUNCT))
    return set(w for w in s.split() if w not in STOP and len(w) > 1)


def pool_stats(trajs, rank_key='g_content'):
    """(mean pairwise jaccard, selected F1, greedy F1) for one sample pool."""
    from bon_grounding import rank_value
    preds = [t['pred'] for t in trajs]
    sets = [cwords(p) for p in preds]
    js = []
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            u = sets[a] | sets[b]
            js.append(len(sets[a] & sets[b]) / len(u) if u else 1.0)
    sel = max(range(len(trajs)),
              key=lambda k: rank_value(trajs[k]['grounding_scores'], rank_key))
    return float(np.mean(js)), sel, 0


def main():
    res = {}

    def analyse_pool(name, path, gt_of=None):
        rows = load_jsonl(path)
        xs, ys = [], []
        for r in rows:
            trajs = r.get('trajectories')
            if not trajs or len(trajs) < 2:
                continue
            j, sel, _ = pool_stats(trajs)
            gt = r['gt']
            f_sel = token_f1(trajs[sel]['pred'], gt) * 100
            f_gre = token_f1(trajs[0]['pred'], gt) * 100
            xs.append(j)
            ys.append(f_sel - f_gre)
        xs, ys = np.array(xs), np.array(ys)
        lr = stats.linregress(xs, ys)
        # quartile dose curve
        q = np.argsort(xs)
        quart = [float(ys[q[i:len(ys) * (i + 1) // 4 or len(ys)]].mean())
                 for i in range(4)] if len(xs) >= 8 else None
        res[name] = dict(
            n=len(xs),
            jaccard_mean=round(float(xs.mean()), 3),
            gain_mean=round(float(ys.mean()), 2),
            slope=round(float(lr.slope), 2), r=round(float(lr.rvalue), 3),
            p=round(float(lr.pvalue), 4),
            gain_quartiles_by_jaccard=[round(x, 2) for x in quart] if quart else None,
            pearson_p=round(float(stats.pearsonr(xs, ys).pvalue), 4))
        return xs, ys

    X, Y, TAG = [], [], []
    for name, path in [
            ('ragtruth_v1_temp_pool', 'outputs/ragtruth_bon_grounding/ragtruth_bon_grounding.jsonl'),
            ('haloquest_v1_temp_pool', 'outputs/haloquest_bon_grounding/haloquest_bon_grounding.jsonl'),
            ('haloquest_v2_temp_pool(81)', 'outputs/haloquest_bon_stacked/haloquest_bon_stacked_full.jsonl')]:
        xs, ys = analyse_pool(name, path)
        X.append(xs); Y.append(ys); TAG.append([name] * len(xs))

    # avhbench window pool (evidence variance): shard files with window candidates
    wx, wy = [], []
    for p in sorted([f for f in os.listdir('outputs/avhbench_bon_window')
                     if f.startswith('avh_probe_shard') and f.endswith('.jsonl')]):
        for line in open(os.path.join('outputs/avhbench_bon_window', p)):
            r = json.loads(line)
            cands = r.get('candidates') or []
            if len(cands) < 2:
                continue
            sets = [cwords(c['pred']) for c in cands]
            js = []
            for a in range(len(sets)):
                for b in range(a + 1, len(sets)):
                    u = sets[a] | sets[b]
                    js.append(len(sets[a] & sets[b]) / len(u) if u else 1.0)
            f1s = [c['f1'] * 100 for c in cands]     # cands[0] is the full-video
            wx.append(float(np.mean(js)))
            wy.append(float(max(f1s[1:]) - f1s[0]))
    if wx:
        lr = stats.linregress(wx, wy)
        res['avhbench_window_pool(oracle ceiling)'] = dict(
            n=len(wx), jaccard_mean=round(float(np.mean(wx)), 3),
            gain_mean=round(float(np.mean(wy)), 2),
            slope=round(float(lr.slope), 2), r=round(float(lr.rvalue), 3),
            p=round(float(lr.pvalue), 4),
            note='y = oracle-window F1 - full-video F1 (selection ceiling), '
                 'evidence-variance pool')

    # pooled within-pool relation across temp pools
    x = np.concatenate(X); y = np.concatenate(Y)
    lr = stats.linregress(x, y)
    res['pooled_temp_pools'] = dict(
        n=len(x), slope=round(float(lr.slope), 2), r=round(float(lr.rvalue), 3),
        p=round(float(lr.pvalue), 4),
        spearman=round(float(stats.spearmanr(x, y).statistic), 3),
        spearman_p=round(float(stats.spearmanr(x, y).pvalue), 4))

    # ---- signal-quality alignment per pool (the second factor of the law) ----
    def alignment(name, paths, score_key='g_content', is_window=False):
        rhos = []
        for path in paths:
            for r in load_jsonl(path):
                if is_window:
                    cands = r.get('candidates') or []
                    if len(cands) < 3:
                        continue
                    sc = [c.get(score_key) for c in cands]
                    f1 = [c['f1'] for c in cands]
                else:
                    trajs = r.get('trajectories')
                    if not trajs or len(trajs) < 3:
                        continue
                    sc = [(t['grounding_scores'] or {}).get(score_key) for t in trajs]
                    f1 = [token_f1(t['pred'], r['gt']) for t in trajs]
                if any(s is None for s in sc):
                    continue
                try:
                    rho = stats.spearmanr(sc, f1).statistic
                except Exception:
                    continue
                if rho == rho:
                    rhos.append(rho)
        res[f'alignment_{name}'] = dict(
            n=len(rhos), mean_within_pool_spearman=round(float(np.mean(rhos)), 3) if rhos else None,
            frac_positive=round(float(np.mean([r > 0 for r in rhos])), 3) if rhos else None,
            note='per-sample Spearman(grounding score, candidate F1) within pool')

    alignment('ragtruth_v1_temp_pool', ['outputs/ragtruth_bon_grounding/ragtruth_bon_grounding.jsonl'])
    alignment('haloquest_v1_temp_pool', ['outputs/haloquest_bon_grounding/haloquest_bon_grounding.jsonl'])
    alignment('haloquest_v2_temp_pool(81)', ['outputs/haloquest_bon_stacked/haloquest_bon_stacked_full.jsonl'])
    alignment('avhbench_window_pool',
              [os.path.join('outputs/avhbench_bon_window', f)
               for f in sorted(os.listdir('outputs/avhbench_bon_window'))
               if f.startswith('avh_probe_shard') and f.endswith('.jsonl')],
              is_window=True)

    with open(OUT, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
