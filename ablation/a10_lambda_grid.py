#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A10 (ABLATION_PLAN.md): lambda x view-set sensitivity grid on CMM (+ PhD).

Pre-registered design: cmm, lambda in {0.25,0.5,1,2,4} x view set
{audio-only, video-only, both}; claim under test: the single exposed
hyperparameter lambda is stable within one order of magnitude, no per-set
tuning needed. Judgment: report the full grid (F1 / Youden), the plateau
width around the shipped lambda, and whether the shipped choice (1.0, both)
is inside the flat region.

Offline simulation is exact here: cmm 720/720 and phd 901/901 online/offline
forced-token identity were validated in the shipped experiments and
re-validated in a1_offline.py (76.36 == 76.36). Zero GPU.
"""
import os
import sys
import json
import numpy as np
from sklearn.metrics import f1_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import load_jsonl

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'a10_lambda_grid.json')


def metrics(yt, yp):
    f1 = f1_score(yt, yp, zero_division=0) * 100
    tpr = recall_score(yt, yp, pos_label=1, zero_division=0)
    tnr = recall_score(yt, yp, pos_label=0, zero_division=0)
    return round(f1, 2), round((tpr + tnr - 1), 3)


def main():
    res = {}
    cmm = load_jsonl('outputs/cf_probe/cmm_probe_full.jsonl')
    grid = {}
    for lam in [0.25, 0.5, 1.0, 2.0, 4.0]:
        for vw, (wa, wv) in [('both(1,1)', (1, 1)), ('audio(1,0)', (1, 0)),
                             ('video(0,1)', (0, 1))]:
            yt, yp = [], []
            for r in cmm:
                d = wa * r['d_cf_audio'] + wv * r['d_cf_video']
                m = r['margin_full'] + lam * d
                yt.append(r['gt_yes'])
                yp.append(1 if m > 0 else 0)
            f1, youden = metrics(yt, yp)
            grid[f'lam={lam}|{vw}'] = dict(F1=f1, Youden=youden)
    res['cmm_grid'] = grid

    # plateau analysis on the shipped view set
    both = {lam: grid[f'lam={lam}|both(1,1)']['F1']
            for lam in [0.25, 0.5, 1.0, 2.0, 4.0]}
    best_lam = max(both, key=both.get)
    within1 = [lam for lam, v in both.items() if v >= both[best_lam] - 1.0]
    res['cmm_plateau_both'] = dict(
        f1_by_lambda={str(k): v for k, v in both.items()},
        best_lambda=best_lam, best_f1=both[best_lam],
        lambdas_within_1pt_of_best=sorted(within1),
        spread_max_min=round(max(both.values()) - min(both.values()), 2),
        shipped_lambda=1.0, shipped_f1_online=76.36,
        shipped_view='both')

    # phd lambda grid on the shipped noctx view (secondary, free)
    phd = load_jsonl('outputs/cf_probe/phd_probe_full.jsonl')
    pgrid = {}
    for lam in [0.25, 0.5, 1.0, 2.0, 4.0]:
        yt, yp = [], []
        for r in phd:
            m = r['margin_full'] + lam * (r['lp_yes_noctx'] - r['lp_yes_full'])
            yt.append(r['gt_yes'])
            yp.append(1 if m > 0 else 0)
        f1, youden = metrics(yt, yp)
        pgrid[f'lam={lam}'] = dict(F1=f1, Youden=youden)
    res['phd_grid_noctx'] = pgrid
    pboth = {float(k.split('=')[1]): v['F1'] for k, v in pgrid.items()}
    pbest = max(pboth, key=pboth.get)
    res['phd_plateau'] = dict(best_lambda=pbest, best_f1=pboth[pbest],
                              f1_by_lambda={str(k): v for k, v in pboth.items()},
                              lambdas_within_1pt=sorted(
                                  [l for l, v in pboth.items() if v >= pboth[pbest] - 1.0]))

    with open(OUT, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
