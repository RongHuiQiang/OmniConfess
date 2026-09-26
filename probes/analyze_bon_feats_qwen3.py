# -*- coding: utf-8 -*-
"""analyze_bon_feats_qwen3.py — 阶段3 特征分析（离线,CPU）。

1. sanity AUC：候选级 VAS/CE 判别力（label = f1 高于 item 内中位数），对照 Q2.5 halo 0.572。
2. 选择增益：各单一特征 & 组合分(grounding Δ + VAS + CE) 选候选的均值 Token-F1，
   对照 greedy(rollout 0) 与随机。
"""
import json, sys, itertools
import numpy as np


def load(path):
    return json.load(open(path))['rows']


def auc(xs, ys):
    pos = [x for x, y in zip(xs, ys) if y == 1]
    neg = [x for x, y in zip(xs, ys) if y == 0]
    if not pos or not neg:
        return None
    s = 0.0
    for a in pos:
        for b in neg:
            s += 1 if a > b else (0.5 if a == b else 0)
    return s / (len(pos) * len(neg))


def feat_get(c, key):
    if key.startswith('vas'):
        kind = 'vas_all' if key.endswith('all') else 'vas_content'
        li = key.split('_')[1]
        v = c['VAS'].get(li)
        return None if v is None else v.get(kind)
    if key in c['CE']:
        return c['CE'][key]
    if key in c:
        return c[key]
    return None


def zrank(vals):
    """rank-normalize to z-scores, None-safe per item."""
    v = [x for x in vals if x is not None]
    if not v:
        return [0.0] * len(vals)
    order = sorted(v)
    med = order[len(order) // 2]
    sd = np.std(v) or 1.0
    return [((x - med) / sd) if x is not None else 0.0 for x in vals]


def main(path):
    rows = load(path)
    print(f'n_items={len(rows)} n_cands={sum(len(r["candidates"]) for r in rows)}')

    feats = ['vas_15_all', 'vas_24_all', 'vas_39_all', 'vas_15_content',
             'vas_24_content', 'vas_39_content', 'CE_top5mean', 'CE_max',
             'g_content', 'g_mean_claim']
    # 1. sanity AUC (per-item median split labels, pooled AUC over within-item pairs)
    print('\n== sanity AUC (within-item pairwise, label = higher f1) ==')
    for f in feats:
        wins = ties = n = 0
        for r in rows:
            cs = r['candidates']
            for a, b in itertools.combinations(cs, 2):
                if a['f1'] == b['f1']:
                    continue
                hi, lo = (a, b) if a['f1'] > b['f1'] else (b, a)
                xh, xl = feat_get(hi, f), feat_get(lo, f)
                if xh is None or xl is None:
                    continue
                n += 1
                if xh > xl:
                    wins += 1
                elif xh == xl:
                    ties += 1
        if n:
            print(f'  {f:16s} AUC={((wins + 0.5 * ties) / n):.4f}  n_pairs={n}')

    # 2. selection gain
    def mean_f1(scorer):
        fs = []
        for r in rows:
            cs = r['candidates']
            if len(cs) <= 1:
                fs.append(cs[0]['f1'] if cs else 0)
                continue
            sc = scorer(cs)
            k = int(np.argmax(sc))
            fs.append(cs[k]['f1'])
        return 100 * np.mean(fs)

    def greedy(cs):
        return [0.0 if not c['sampled'] else -1.0 for c in cs]

    def single(f):
        def s(cs):
            return zrank([feat_get(c, f) for c in cs])
        return s

    def combo(weights):
        def s(cs):
            zs = {f: zrank([feat_get(c, f) for c in cs]) for f in weights}
            return [sum(weights[f] * zs[f][k] for f in weights)
                    for k in range(len(cs))]
        return s

    print('\n== selection mean Token-F1 ==')
    print(f'  greedy            {mean_f1(greedy):.2f}')
    rng = np.random.default_rng(0)
    rnd = np.mean([mean_f1(lambda cs, s=rng.random(): [s + k * 1e-9 for k in range(len(cs))]) for _ in range(5)])
    print(f'  random(~5x)       {rnd:.2f}')
    for f in feats:
        print(f'  sel_{f:15s} {mean_f1(single(f)):.2f}')
    print(f'  oracle(max f1)    {100*np.mean([max(c["f1"] for c in r["candidates"]) for r in rows]):.2f}')
    for w in [
        {'g_content': 1.0, 'vas_24_all': 1.0},
        {'g_content': 1.0, 'vas_24_content': 1.0},
        {'g_content': 1.0, 'CE_top5mean': -1.0},
        {'g_content': 1.0, 'vas_24_all': 1.0, 'CE_top5mean': -1.0},
        {'vas_24_all': 1.0, 'CE_top5mean': -1.0},
    ]:
        print(f'  combo {w}  {mean_f1(combo(w)):.2f}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'outputs/bon_probe_qwen3/feat_full.json')
