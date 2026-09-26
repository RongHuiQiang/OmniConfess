# -*- coding: utf-8 -*-
"""lambda_sweep.py -- P7: exact offline lambda full-spectrum sweep for the
three judgement-family datasets (phd / cmm / pubmedqa).

Source: outputs/cf_probe/{phd,cmm,pubmedqa}_probe_full.jsonl (full-set probes
with per-sample decision-token logprobs, counterfactual deltas, gt label and
base prediction). The decision is a deterministic function of these recorded
quantities, so the sweep is exact for step-0 (or recorded-decision-step)
interventions -- same precedent as cf_probe_analyze_judgement.py and the
phd section-4 offline==online check.

Formulas (line-checked against the operators):
  phd (cf_confess_phd.py L111-118):
      margin_full = lp_yes_full - lp_no_full ; delta = lp_yes_cf - lp_yes_full
      forced = Yes iff margin_full + lam*delta > 0
      probe CF view (image+question, context removed) == 'noctx' columns.
  cmm (cf_confess_cmm.py, doc L781):
      forced = Yes iff margin_full + lam*d_cf_sum > 0
      (pre-registered equal weights w_a = w_v = 1).
  pubmedqa (cf_confess_pubmedqa.py L109-117):
      yes_adj = lp_yes_full + lam*d_cf
      pr = yes if yes_adj > max(lp_no_full, lp_maybe_full) else (no if
      lp_no_full > lp_maybe_full else maybe); maybe pred counts as miss in
      yes-F1 (identical to cf_probe_analyze_judgement.sim_pubmedqa).

Outputs per dataset (one json per dataset, incremental on disk):
  trigger_rate, F1(yes), youden, yes_rate, changed/fix/destroy counts per lam,
  plus base (lam=0) reference.

Usage:
  python lambda_sweep.py phd   [out.json]
  python lambda_sweep.py cmm   [out.json]
  python lambda_sweep.py pubmedqa [out.json]
"""
import json, sys, os
import numpy as np
from sklearn.metrics import f1_score, recall_score

PROBE = '/home/tsq/rhq/omniconfess/OmniConfess/outputs/cf_probe'
DEF_OUT = '/home/tsq/rhq/omniconfess/OmniConfess/outputs/problem_structure_fig'

LAM_STAR = {'phd': 2.0, 'cmm': 1.0, 'pubmedqa': 0.25}
GRIDS = {
    'phd': [-1.0, 0.0, 0.25, 0.5, 1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0, 6.0],
    'cmm': [-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0],
    'pubmedqa': [-1.0, 0.0, 0.1, 0.25, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
}


def load(ds):
    p = os.path.join(PROBE, f'{ds}_probe_full.jsonl')
    return [json.loads(l) for l in open(p)]


def sweep_phd(rows, lams):
    m = np.array([r['margin_full'] for r in rows], float)
    d = np.array([r['lp_yes_noctx'] - r['lp_yes_full'] for r in rows], float)
    gt = np.array([r['gt_yes'] for r in rows], int)
    base = np.array([1 if str(r['pred']).strip().lower() == 'yes' else 0
                     for r in rows], int)
    return _binary_sweep(m, d, gt, base, lams)


def sweep_cmm(rows, lams):
    m = np.array([r['margin_full'] for r in rows], float)
    d = np.array([r['d_cf_sum'] for r in rows], float)
    gt = np.array([r['gt_yes'] for r in rows], int)
    base = np.array([1 if str(r['pred']).strip().lower() == 'yes' else 0
                     for r in rows], int)
    return _binary_sweep(m, d, gt, base, lams)


def _binary_sweep(m, d, gt, base, lams):
    out = []
    for lam in lams:
        pr = (m + lam * d > 0).astype(int)
        changed = pr != base
        f1 = f1_score(gt, pr, zero_division=0)
        tpr = recall_score(gt, pr, pos_label=1, zero_division=0)
        tnr = recall_score(gt, pr, pos_label=0, zero_division=0)
        out.append(dict(
            lam=lam, n=int(len(gt)),
            trigger_rate=round(float(changed.mean()), 4),
            changed=int(changed.sum()),
            fix=int((changed & (pr == gt)).sum()),
            destroy=int((changed & (pr != gt)).sum()),
            f1=round(f1 * 100, 2),
            youden=round(tpr - (1 - tnr), 3),
            yes_rate=round(float(pr.mean()) * 100, 2),
        ))
    return out, dict(
        n=int(len(gt)), base_f1=round(f1_score(gt, base, zero_division=0) * 100, 2),
        base_yes_rate=round(float(base.mean()) * 100, 2),
        btpr=round(recall_score(gt, base, pos_label=1, zero_division=0), 3),
        btnr=round(recall_score(gt, base, pos_label=0, zero_division=0), 3))


def sweep_pubmedqa(rows, lams):
    ly = np.array([r['lp_yes_full'] for r in rows], float)
    ln = np.array([r['lp_no_full'] for r in rows], float)
    lm = np.array([r['lp_maybe_full'] for r in rows], float)
    d = np.array([r['d_cf'] for r in rows], float)
    gt = np.array([r['gt_yes'] for r in rows], int)
    base = np.array([r['pred_tok'] if r['pred_tok'] in (0, 1) else 2
                     for r in rows], int)
    out = []
    for lam in lams:
        ya = ly + lam * d
        pr = np.where(ya > np.maximum(ln, lm), 1,
                      np.where(ln > lm, 0, 2)).astype(int)
        pb = np.where(pr == 2, 1 - gt, pr)      # maybe pred = miss (sim identical)
        changed = pr != base
        f1 = f1_score(gt, pb, zero_division=0)
        tpr = recall_score(gt, pb, pos_label=1, zero_division=0)
        tnr = recall_score(gt, pb, pos_label=0, zero_division=0)
        out.append(dict(
            lam=lam, n=int(len(gt)),
            trigger_rate=round(float(changed.mean()), 4),
            changed=int(changed.sum()),
            fix=int((changed & (pb == gt)).sum()),
            destroy=int((changed & (pb != gt)).sum()),
            f1=round(f1 * 100, 2),
            youden=round(tpr - (1 - tnr), 3),
            yes_rate=round(float((pb == 1).mean()) * 100, 2),
        ))
    bpb = np.where(base == 2, 1 - gt, base)
    return out, dict(
        n=int(len(gt)),
        base_f1=round(f1_score(gt, bpb, zero_division=0) * 100, 2),
        base_yes_rate=round(float((bpb == 1).mean()) * 100, 2),
        btpr=round(recall_score(gt, bpb, pos_label=1, zero_division=0), 3),
        btnr=round(recall_score(gt, bpb, pos_label=0, zero_division=0), 3))


def main():
    ds = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(DEF_OUT, f'lambda_sweep_{ds}.json')
    rows = load(ds)
    lams = GRIDS[ds]
    if ds == 'phd':
        grid, base = sweep_phd(rows, lams)
    elif ds == 'cmm':
        grid, base = sweep_cmm(rows, lams)
    else:
        grid, base = sweep_pubmedqa(rows, lams)
    res = dict(dataset=ds, lam_star=LAM_STAR[ds], source=os.path.join(
        PROBE, f'{ds}_probe_full.jsonl'), base=base, grid=grid)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=1)
    print(f'[{ds}] n={base["n"]} base_f1={base["base_f1"]} -> {out_path}')
    for g in grid:
        print('  ', g)


if __name__ == '__main__':
    main()
