#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""probe_judgment_phd_nemotron.py -- pre-registered judgment for the
phd x Nemotron CF probe (probe_cf_signal_phd_nemotron.py). Written BEFORE the
probe ran; implements the conjunction in that file's header verbatim:

  ALIVE := AUC(-delta_noctx -> gt) > 0.65
           AND (i)  mean delta_noctx > 0 on {gt=no, base=yes}   (FP mass)
           AND (ii) mean delta_noctx < 0 on {gt=yes, base=no}   (FN mass)
           AND (iii) median |margin_full| error rows < median correct rows

Supplementary (reported, not gate-relevant): bootstrap CI, all-signal AUCs,
cell means, offline lambda grid (operator: margin' = margin_full +
lam * d_noctx > 0 -> Yes), tf_repro distribution.

Usage: python probe_judgment_phd_nemotron.py outputs/cf_probe/phd_probe_nemotron.jsonl
"""
import json, sys
import numpy as np
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
          f'argmax==dec {sum(r["full_argmax_is_dec"] for r in rows)}/{len(rows)})')

    gt = [r['gt_yes'] for r in rows]
    print('\n== AUC -> gt ==')
    aucs = {}
    for k, sign in [('margin_full', 1), ('-delta_noctx', -1),
                    ('delta_noctx', 1), ('-delta_noimg', -1),
                    ('margin_noctx', 1)]:
        a, ci = auc_ci([sign * r[k.lstrip('-')] for r in rows], gt)
        aucs[k] = a
        print(f'  {k:16s} AUC={a:.3f} [{ci[0]:.3f},{ci[1]:.3f}]')

    print('\n== direction clauses (pre-registered) ==')
    fp = [r for r in rows if r['gt_yes'] == 0 and r['pred'] == 'yes']
    fn = [r for r in rows if r['gt_yes'] == 1 and r['pred'] == 'no']
    err = [r for r in rows if not r['base_correct']]
    cor = [r for r in rows if r['base_correct']]
    c1 = np.mean([r['delta_noctx'] for r in fp]) if fp else float('nan')
    c2 = np.mean([r['delta_noctx'] for r in fn]) if fn else float('nan')
    m_err = np.median([abs(r['margin_full']) for r in err])
    m_cor = np.median([abs(r['margin_full']) for r in cor])
    print(f'  (i)  FP(gt=no,base=yes) n={len(fp)} mean delta_noctx={c1:+.3f} '
          f'-> {"PASS" if c1 > 0 else "FAIL"}')
    print(f'  (ii) FN(gt=yes,base=no) n={len(fn)} mean delta_noctx={c2:+.3f} '
          f'-> {"PASS" if c2 < 0 else "FAIL"}')
    print(f'  (iii) |margin| median error {m_err:.3f} vs correct {m_cor:.3f} '
          f'-> {"PASS" if m_err < m_cor else "FAIL"}')

    print('\n== cell means (gt x base-pred), Qwen2.5-style table ==')
    for gy in (1, 0):
        for pr in ('yes', 'no'):
            sub = [r for r in rows if r['gt_yes'] == gy and r['pred'] == pr]
            if not sub:
                continue
            print(f'  gt={"yes" if gy else "no":3s} base={pr:3s} n={len(sub):3d} '
                  f'margin={np.mean([r["margin_full"] for r in sub]):+.3f} '
                  f'd_noctx={np.mean([r["delta_noctx"] for r in sub]):+.3f} '
                  f'|d|med={np.median([abs(r["delta_noctx"]) for r in sub]):.4f}')
    m = np.array([r['margin_full'] for r in rows])
    d = np.array([r['d_noctx'] for r in rows])
    print(f'  flippable band |m|<1*|d|: {(np.abs(m) < np.abs(d)).mean()*100:.1f}% '
          f'(Qwen2.5 ~25%, Qwen3 13%)')

    print('\n== offline lambda grid (exact at the recorded decision step) ==')
    g = np.array(gt)
    base = np.array([1 if r['pred'] == 'yes' else 0 for r in rows])
    for lam in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
        pr = (m + lam * d > 0).astype(int)
        tpr = recall_score(g, pr, pos_label=1, zero_division=0)
        tnr = recall_score(g, pr, pos_label=0, zero_division=0)
        print(f'  lam={lam:<5} F1={f1_score(g, pr, zero_division=0)*100:6.2f} '
              f'youden={tpr-(1-tnr):+.3f} yes={pr.mean()*100:5.1f}% '
              f'(base {base.mean()*100:.1f}%) changed={int((pr != base).sum())}')

    alive = (aucs['-delta_noctx'] > 0.65) and (c1 > 0) and (c2 < 0) and (m_err < m_cor)
    print(f'\n== VERDICT (pre-registered conjunction): '
          f'{"ALIVE -> full 906 CF@Nemotron-phd" if alive else "DEAD -> self-healing third point"} ==')


if __name__ == '__main__':
    main()
