#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""xbackbone_cell_compare.py -- [跨骨干矩阵] cell report: prompt arm vs base
on the same backbone, judgement datasets (Yes/No[/Maybe] metrics).

Reads two *_single_scored.jsonl files (arm, base), pairs by id, reports
F1 / accuracy / yes-rate / recall split (using the official scorer's
eval_metrics.Prediction / Ground_Truth codes: 1=yes 0=no 2=maybe) and an
exact McNemar on the binary decision. The headline CSV numbers (Yes-F1,
PhD-Index) still come from single_eval; this adds the paired view.

Usage: python xbackbone_cell_compare.py <arm.jsonl> <base.jsonl>
"""
import json, sys
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import binomtest


def load(p):
    rows = [json.loads(l) for l in open(p)]
    return {r['id']: r for r in rows if r.get('eval_metrics')}


def main():
    arm_p, base_p = sys.argv[1], sys.argv[2]
    arm, base = load(arm_p), load(base_p)
    ids = [i for i in arm if i in base]
    print(f'arm={len(arm)} base={len(base)} paired={len(ids)}')

    def vec(d):
        gt = np.array([d[i]['eval_metrics']['Ground_Truth'] for i in ids])
        pr = np.array([d[i]['eval_metrics']['Prediction'] for i in ids])
        return gt, pr

    ga, pa = vec(arm)
    gb, pb = vec(base)

    def report(name, g, p):
        binary = p != 2
        acc = (g == p).mean()                      # maybe==gt never true
        f1 = f1_score(g[binary & (g != 2)], p[binary & (g != 2)],
                      zero_division=0) if (binary & (g != 2)).any() else float('nan')
        r_yes = ((p == 1) & (g == 1)).sum() / max((g == 1).sum(), 1)
        r_no = ((p == 0) & (g == 0)).sum() / max((g == 0).sum(), 1)
        print(f'  {name}: acc={acc*100:.2f} binF1={f1*100:.2f} '
              f'yes_rate={(p==1).mean()*100:.1f}% maybe={(p==2).sum()} '
              f'r_yes={r_yes*100:.2f} r_no={r_no*100:.2f}')

    report('arm', ga, pa)
    report('base', gb, pb)

    # McNemar on binary accuracy over rows where both arms gave a binary pred
    m = (pa != 2) & (pb != 2)
    a_ok = (pa == ga)[m]
    b_ok = (pb == gb)[m]
    n01 = int((a_ok & ~b_ok).sum())
    n10 = int((~a_ok & b_ok).sum())
    p_mcn = binomtest(n01, n01 + n10).pvalue if (n01 + n10) else float('nan')
    print(f'  binary-paired n={int(m.sum())}: fixed={n01} broken={n10} '
          f'McNemar p={p_mcn:.3g}')
    # where the maybe mass moved
    if (pb == 2).any() or (pa == 2).any():
        from collections import Counter
        moves = Counter((pb[i], pa[i]) for i in range(len(ids)))
        print('  maybe-relevant transitions base->arm:',
              {k: v for k, v in moves.items() if k[0] == 2 or k[1] == 2})


if __name__ == '__main__':
    main()
