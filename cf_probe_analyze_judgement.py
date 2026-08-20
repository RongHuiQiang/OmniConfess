#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cf_probe_analyze_judgement.py -- analysis for the cmm / pubmedqa CF probes.

Reads probe JSONL(s) (dev or full; pass multiple files to concat shards) and
reports:
  - AUC (+ bootstrap 95% CI) of each signal for predicting gt, overall and
    in the two fix-direction subgroups (base-pred yes / base-pred no);
  - directionality: mean signal per (gt x base-pred) cell;
  - cmm: per-subgroup |Delta| stats and group separation (MWU);
  - offline lambda-grid simulation on the probe records. For step-0 bare
    decisions the operator's intervention is an exact deterministic function
    of the recorded quantities (margin_full, d_cf*), so the grid is exact
    given bf16 determinism (phd section 4 precedent: offline sim == online).

Usage:
  python cf_probe_analyze_judgement.py cmm outputs/cf_probe/cmm_probe.jsonl
  python cf_probe_analyze_judgement.py pubmedqa outputs/cf_probe/pubmedqa_probe.jsonl
"""
import json, sys, argparse, random
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score
from scipy.stats import mannwhitneyu


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p)]
    return rows


def auc_ci(x, y, n=2000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, int)
    if len(set(y)) < 2:
        return float('nan'), (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    aucs = []
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(set(y[s])) < 2:
            continue
        aucs.append(roc_auc_score(y[s], x[s]))
    return roc_auc_score(y, x), (float(np.percentile(aucs, 2.5)),
                                 float(np.percentile(aucs, 97.5)))


def sim_cmm(rows, lams, variant='sum'):
    """Exact offline simulation (decision = step 0 bare for all probe rows).
    variant: 'sum' = w_a*d_audio + w_v*d_video (pre-registered equal weights);
             'audio' = d_cf_audio only; 'video' = d_cf_video only."""
    out = []
    m = np.array([r['margin_full'] for r in rows])
    if variant == 'sum':
        d = np.array([r['d_cf_sum'] for r in rows])
    elif variant == 'audio':
        d = np.array([r['d_cf_audio'] for r in rows])
    elif variant == 'video':
        d = np.array([r['d_cf_video'] for r in rows])
    else:
        raise ValueError(variant)
    gt = np.array([r['gt_yes'] for r in rows])
    base = np.array([1 if r['pred'] == 'yes' else 0 for r in rows])
    for lam in lams:
        pr = (m + lam * d > 0).astype(int)
        f1 = f1_score(gt, pr, zero_division=0)
        tpr = recall_score(gt, pr, pos_label=1, zero_division=0)
        fpr = recall_score(gt, pr, pos_label=0, zero_division=0)
        out.append(dict(lam=lam, f1=round(f1 * 100, 2),
                        youden=round(tpr - (1 - fpr), 3),
                        tpr=round(tpr, 3), tnr=round(fpr, 3),
                        yes_rate=round(pr.mean() * 100, 2),
                        changed=int((pr != base).sum())))
    return out


def sim_pubmedqa(rows, lams):
    out = []
    ly = np.array([r['lp_yes_full'] for r in rows])
    ln = np.array([r['lp_no_full'] for r in rows])
    lm = np.array([r['lp_maybe_full'] for r in rows])
    d = np.array([r['d_cf'] for r in rows])
    gt = np.array([r['gt_yes'] for r in rows])          # 1/0, maybe excluded
    base = np.array([r['pred_tok'] if r['pred_tok'] in (0, 1) else 2
                     for r in rows])
    for lam in lams:
        ya = ly + lam * d
        # class: yes if adjusted-yes beats no & maybe; else no vs maybe
        pr = np.where(ya > np.maximum(ln, lm), 1,
                      np.where(ln > lm, 0, 2)).astype(int)
        pb = np.where(pr == 2, 1 - gt, pr)              # maybe pred = miss
        f1 = f1_score(gt, pb, zero_division=0)
        tpr = recall_score(gt, pb, pos_label=1, zero_division=0)
        tnr = recall_score(gt, pb, pos_label=0, zero_division=0)
        out.append(dict(lam=lam, f1=round(f1 * 100, 2),
                        youden=round(tpr - (1 - tnr), 3),
                        tpr=round(tpr, 3), tnr=round(tnr, 3),
                        yes_rate=round((pb == 1).mean() * 100, 2),
                        changed=int((pr != base).sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=['cmm', 'pubmedqa'])
    ap.add_argument('probe_files', nargs='+')
    args = ap.parse_args()

    rows = load(args.probe_files)
    print(f'n={len(rows)}')

    if args.dataset == 'cmm':
        gt = [r['gt_yes'] for r in rows]
        sig = {
            'margin_full': [r['margin_full'] for r in rows],
            'delta_audio(full-masked)': [r['delta_audio'] for r in rows],
            'delta_video(full-masked)': [r['delta_video'] for r in rows],
            '-d_cf_sum': [-r['d_cf_sum'] for r in rows],
            'd_cf_sum': [r['d_cf_sum'] for r in rows],
            'margin_noaudio': [r['margin_noaudio'] for r in rows],
            'margin_novideo': [r['margin_novideo'] for r in rows],
        }
    else:
        gt = [r['gt_yes'] for r in rows]
        sig = {
            'margin_full': [r['margin_full'] for r in rows],
            'delta_passage(full-masked)': [r['delta_passage'] for r in rows],
            'd_cf': [r['d_cf'] for r in rows],
            'margin_nopassage': [r['margin_nopassage'] for r in rows],
        }

    print('\n== AUC -> gt (all probe rows) ==')
    for k, v in sig.items():
        a, ci = auc_ci(v, gt)
        print(f'  {k:32s} AUC={a:.3f} [{ci[0]:.3f},{ci[1]:.3f}]')

    # fix-direction subgroups
    if args.dataset == 'cmm':
        for name, sel, key in [
                ('base-pred=yes -> gt=no', lambda r: r['pred'] == 'yes', 'gt'),
                ('base-pred=no  -> gt=yes', lambda r: r['pred'] == 'no', 'gt')]:
            sub = [r for r in rows if sel(r)]
            yy = [1 - r['gt_yes'] for r in sub] if name.startswith('base-pred=yes') \
                else [r['gt_yes'] for r in sub]
            print(f'\n== {name} (n={len(sub)}) ==')
            for k in ('delta_audio', 'delta_video', 'd_cf_sum'):
                a, ci = auc_ci([r[k] for r in sub], yy)
                print(f'  {k:14s} AUC={a:.3f} [{ci[0]:.3f},{ci[1]:.3f}]')
        print('\n== subgroup |Delta| (cmm facets) ==')
        for g in sorted(set(r['group'] for r in rows)):
            sub = [r for r in rows if r['group'] == g]
            print(f'  {g}: n={len(sub)} '
                  f'|d_audio| med={np.median([abs(r["d_cf_audio"]) for r in sub]):.3f} '
                  f'mean={np.mean([abs(r["d_cf_audio"]) for r in sub]):.3f} | '
                  f'|d_video| med={np.median([abs(r["d_cf_video"]) for r in sub]):.3f} '
                  f'mean={np.mean([abs(r["d_cf_video"]) for r in sub]):.3f}')
        for k in ('d_cf_audio', 'd_cf_video'):
            a = [abs(r[k]) for r in rows if r['group'].startswith('overrely_audio')]
            b = [abs(r[k]) for r in rows if r['group'].startswith('overrely_visual')]
            u, p = mannwhitneyu(a, b, alternative='two-sided')
            print(f'  MWU {k}: audio-grp mean={np.mean(a):.3f} '
                  f'visual-grp mean={np.mean(b):.3f} p={p:.4g}')
        print('\n== cell means (gt x base-pred) ==')
        for gy in (1, 0):
            for pr in ('yes', 'no'):
                sub = [r for r in rows if r['gt_yes'] == gy and r['pred'] == pr]
                if not sub:
                    continue
                print(f'  gt={"yes" if gy else "no":3s} base={pr:3s} n={len(sub):3d} '
                      f'margin={np.mean([r["margin_full"] for r in sub]):+.3f} '
                      f'd_a={np.mean([r["d_cf_audio"] for r in sub]):+.3f} '
                      f'd_v={np.mean([r["d_cf_video"] for r in sub]):+.3f} '
                      f'd_sum={np.mean([r["d_cf_sum"] for r in sub]):+.3f}')
        print('\n== offline lambda grid, variant=sum (exact for step-0 decisions) ==')
        for s in sim_cmm(rows, [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]):
            print(' ', s)
        print('== variant=audio only ==')
        for s in sim_cmm(rows, [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                         variant='audio'):
            print(' ', s)
        print('== variant=video only ==')
        for s in sim_cmm(rows, [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                         variant='video'):
            print(' ', s)
        base_f1 = f1_score([r['gt_yes'] for r in rows],
                           [1 if r['pred'] == 'yes' else 0 for r in rows])
        base_tpr = recall_score([r['gt_yes'] for r in rows],
                                [1 if r['pred'] == 'yes' else 0 for r in rows])
        base_tnr = recall_score([r['gt_yes'] for r in rows],
                                [1 if r['pred'] == 'yes' else 0 for r in rows],
                                pos_label=0)
        print(f'  base dev F1={base_f1*100:.2f} yes_rate='
              f'{np.mean([r["pred"]=="yes" for r in rows])*100:.1f} '
              f'youden={round(base_tpr-(1-base_tnr),3)}')
    else:
        print('\n== cell means (gt x base-pred-token) ==')
        for gy in (1, 0):
            for pt in (1, 0, 2):
                sub = [r for r in rows if r['gt_yes'] == gy and r['pred_tok'] == pt]
                if not sub:
                    continue
                lab = {1: 'yes', 0: 'no', 2: 'maybe'}[pt]
                print(f'  gt={"yes" if gy else "no":3s} base={lab:5s} n={len(sub):3d} '
                      f'margin={np.mean([r["margin_full"] for r in sub]):+.3f} '
                      f'd_cf={np.mean([r["d_cf"] for r in sub]):+.3f}')
        print('\n== offline lambda grid (exact for step-0 decisions); '
              'lambda<0 = CAD-direction (passage-boosting) ==')
        for s in sim_pubmedqa(rows, [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5,
                                     0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]):
            print(' ', s)
        base_pb = np.array([1 - r['gt_yes'] if r['pred_tok'] == 2 else r['pred_tok']
                            for r in rows])
        gt = np.array([r['gt_yes'] for r in rows])
        base_tpr = recall_score(gt, base_pb, pos_label=1)
        base_tnr = recall_score(gt, base_pb, pos_label=0)
        print(f'  base F1={f1_score(gt, base_pb)*100:.2f} '
              f'yes_rate={(base_pb==1).mean()*100:.1f} '
              f'youden={round(base_tpr-(1-base_tnr),3)}')


if __name__ == '__main__':
    main()
