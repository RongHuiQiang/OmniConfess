#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_judgment_cmm_nemotron.py -- pre-registered judgment for the
cmm x Nemotron CF probe (probe_cf_signal_cmm_nemotron.py). Written BEFORE the
probe ran; implements the conjunction in that file's header verbatim (copied
from the cmm-Qwen3 pre-registration):

  ALIVE := AUC(-d_cf_sum -> gt) > 0.65
           AND (i)  share of base-Yes rows with d_cf_sum < 0 is large
                    (disease = a modality is propping Yes; Q2.5 99.9%)
           AND (ii) mean d_cf_sum < 0 on {gt=no, base=yes}   (FP mass)
           AND (iii) mean margin_full on error-Yes rows < on correct-Yes rows

Supplementary (reported, non-gate): all-signal AUCs + CI, cell means,
per-subgroup |d_cf| + MWU, offline lambda grid (margin' = margin_full +
lam * d_cf_sum > 0 -> Yes), tf_repro, magnitude table vs Qwen2.5/Qwen3.

Usage: python probe_judgment_cmm_nemotron.py outputs/cf_probe/cmm_probe_nemotron.jsonl
"""
import json, sys
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, f1_score, recall_score


def auc_ci(x, y, n=2000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, int)
    if len(set(y)) < 2:
        return float('nan'), (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(set(y[s])) < 2:
            continue
        aucs.append(roc_auc_score(y[s], x[s]))
    return roc_auc_score(y, x), (float(np.percentile(aucs, 2.5)),
                                 float(np.percentile(aucs, 97.5)))


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    print(f'n={len(rows)} '
          f'(bare={sum(r["bare"] for r in rows)}, '
          f'tf_repro med={np.median([abs(r["tf_repro"]) for r in rows]):.5f}, '
          f'argmax==dec {sum(r["full_argmax_is_dec"] for r in rows)}/{len(rows)}, '
          f'input_len match {sum(r["input_len_matches"] for r in rows)}/{len(rows)})')

    gt = [r['gt_yes'] for r in rows]
    print('\n== AUC -> gt ==')
    aucs = {}
    for k, sign in [('margin_full', 1), ('-d_cf_sum', -1), ('d_cf_sum', 1),
                    ('delta_audio', 1), ('delta_video', 1),
                    ('margin_noaudio', 1), ('margin_novideo', 1)]:
        a, ci = auc_ci([sign * r[k.lstrip('-')] for r in rows], gt)
        aucs[k] = a
        print(f'  {k:16s} AUC={a:.3f} [{ci[0]:.3f},{ci[1]:.3f}]')

    print('\n== direction clauses (pre-registered) ==')
    byes = [r for r in rows if r['pred'] == 'yes']
    share = (np.array([r['d_cf_sum'] for r in byes]) < 0).mean() if byes else float('nan')
    fp = [r for r in rows if r['gt_yes'] == 0 and r['pred'] == 'yes']
    c2 = np.mean([r['d_cf_sum'] for r in fp]) if fp else float('nan')
    eyes = [r for r in rows if r['pred'] == 'yes' and not r['base_correct']]
    cyes = [r for r in rows if r['pred'] == 'yes' and r['base_correct']]
    m_err = np.mean([r['margin_full'] for r in eyes]) if eyes else float('nan')
    m_cor = np.mean([r['margin_full'] for r in cyes]) if cyes else float('nan')
    print(f'  (i)  base=yes n={len(byes)} share(d_cf_sum<0)={share*100:.1f}% '
          f'-> {"PASS" if share > 0.75 else "FAIL"} (Q2.5 99.9%)')
    print(f'  (ii) FP(gt=no,base=yes) n={len(fp)} mean d_cf_sum={c2:+.3f} '
          f'-> {"PASS" if c2 < 0 else "FAIL"} (Q2.5 -2.43; Qwen3 +0.26 INVERTED)')
    print(f'  (iii) mean margin error-Yes {m_err:+.3f} vs correct-Yes {m_cor:+.3f} '
          f'-> {"PASS" if m_err < m_cor else "FAIL"}')

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

    print('\n== magnitude vs generations (medians) ==')
    print(f'  |d_cf_sum| med={np.median([abs(r["d_cf_sum"]) for r in rows]):.4f} '
          f'(Q2.5 2.258 / Qwen3 0.0037)   |margin| med='
          f'{np.median([abs(r["margin_full"]) for r in rows]):.3f} '
          f'(Q2.5 1.62 / Qwen3 10.12)')
    m = np.array([r['margin_full'] for r in rows])
    d = np.array([r['d_cf_sum'] for r in rows])
    print(f'  flippable band |m|<1*|d|: {(np.abs(m) < np.abs(d)).mean()*100:.1f}% '
          f'(Q2.5 56.4% / Qwen3 6.1%)')
    a = [abs(r['d_cf_audio']) for r in rows if r['group'].startswith('overrely_audio')]
    b = [abs(r['d_cf_audio']) for r in rows if r['group'].startswith('overrely_visual')]
    if a and b:
        u, p = mannwhitneyu(a, b, alternative='two-sided')
        print(f'  MWU |d_audio| audio-grp={np.mean(a):.3f} '
              f'visual-grp={np.mean(b):.3f} p={p:.4g}')

    print('\n== offline lambda grid (exact at the recorded decision step) ==')
    g = np.array(gt)
    base = np.array([1 if r['pred'] == 'yes' else 0 for r in rows])
    for lam in [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        pr = (m + lam * d > 0).astype(int)
        tpr = recall_score(g, pr, pos_label=1, zero_division=0)
        tnr = recall_score(g, pr, pos_label=0, zero_division=0)
        print(f'  lam={lam:<5} F1={f1_score(g, pr, zero_division=0)*100:6.2f} '
              f'youden={tpr-(1-tnr):+.3f} yes={pr.mean()*100:5.1f}% '
              f'(base {base.mean()*100:.1f}%) changed={int((pr != base).sum())}')

    alive = (aucs['-d_cf_sum'] > 0.65) and (share > 0.75) and (c2 < 0) and (m_err < m_cor)
    print(f'\n== VERDICT (pre-registered conjunction): '
          f'{"ALIVE -> full 721 CF@Nemotron-cmm" if alive else "DEAD -> negative result, self-healing program closes"} ==')


if __name__ == '__main__':
    main()
