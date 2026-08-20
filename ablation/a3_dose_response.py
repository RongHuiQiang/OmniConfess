#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A3 (ABLATION_PLAN.md): signal-benefit dose-response curve.

Pre-registered design (plan text):
  5 measured points (signal AUC -> operator F1 gain). Fit the AUC->gain
  relation. Expected: threshold effect, AUC<0.6 -> no gain. This curve is the
  "when to use CF" theoretical predictor, replacing unproven propositions
  with measurement. If the cmm@Qwen3 probe lands, add it as point 6.

Everything here is zero-GPU: AUCs recomputed from the probe jsonl files on
disk, gains recomputed from the *_single_scored files with paired tests.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import (load_jsonl, bootstrap_auc, judgement_preds, f1_yp,
                  paired_mcnemar_from_preds, align_rows, pair_by_question,
                  token_f1_per_row, paired_perm_test)
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'a3_dose_response.json')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def auc_block(rows, signal_fn, name):
    x = [signal_fn(r) for r in rows]
    y = [r['gt_yes'] for r in rows]
    pt, ci = bootstrap_auc(x, y)
    return dict(name=name, n=len(rows), auc=round(pt, 3), ci=[round(ci[0], 3), round(ci[1], 3)])


def gain_block(label, base_file, arm_file, extra_arms=None, randctl_file=None):
    extra_arms = extra_arms or {}
    files = [base_file, arm_file] + ([randctl_file] if randctl_file else []) + list(extra_arms.values())
    try:
        rows = align_rows(*files)
    except ValueError:
        rows = pair_by_question(*files)
    base_rows, arm_rows = rows[0], rows[1]
    rc_rows = rows[2] if randctl_file else None
    other_rows = rows[(3 if randctl_file else 2):]

    gts = [g for g, p in judgement_preds(base_rows)]
    pb = [p for g, p in judgement_preds(base_rows)]
    pa = [p for g, p in judgement_preds(arm_rows)]
    f1_base, f1_arm = f1_yp(gts, pb), f1_yp(gts, pa)
    b, c, p_val = paired_mcnemar_from_preds(pb, pa, gts)   # b: base wrong arm right
    out = dict(label=label, n=len(gts), f1_base=round(f1_base, 2),
               f1_arm=round(f1_arm, 2), gain=round(f1_arm - f1_base, 2),
               fix=b, break_=c, p_mcnemar_vs_base=p_val)
    if rc_rows is not None:
        pr = [p for g, p in judgement_preds(rc_rows)]
        f1_rc = f1_yp(gts, pr)
        b2, c2, p_rc = paired_mcnemar_from_preds(pa, pr, gts)
        out.update(f1_randctl=round(f1_rc, 2),
                   net_gain=round(f1_arm - f1_rc, 2), p_vs_randctl=p_rc)
    for nm, r in zip(extra_arms.keys(), other_rows):
        po = [p for g, p in judgement_preds(r)]
        out[f'f1_{nm}'] = round(f1_yp(gts, po), 2)
    return out


def main():
    res = {}

    # ---------------- probes ----------------
    phd_dev = load_jsonl('outputs/cf_probe/phd_probe.jsonl')
    phd_full = load_jsonl('outputs/cf_probe/phd_probe_full.jsonl')
    cmm_full = load_jsonl('outputs/cf_probe/cmm_probe_full.jsonl')
    pmq_full = load_jsonl('outputs/cf_probe/pubmedqa_probe_full.jsonl')
    phd_q3 = load_jsonl('outputs/cf_probe/phd_probe_qwen3.jsonl')

    aucs = {}
    # phd operator signal = lp_yes_noctx - lp_yes_full  (context suppression of Yes)
    aucs['phd_q2.5_dev'] = auc_block(phd_dev, lambda r: r['lp_yes_noctx'] - r['lp_yes_full'], 'd_cf(noctx) dev100')
    aucs['phd_q2.5_full'] = auc_block(phd_full, lambda r: r['lp_yes_noctx'] - r['lp_yes_full'], 'd_cf(noctx) full906')
    aucs['phd_q2.5_full_noimg'] = auc_block(phd_full, lambda r: r['lp_yes_noimg'] - r['lp_yes_full'], 'd_cf(noimg) full906')
    aucs['cmm_q2.5_full'] = auc_block(cmm_full, lambda r: -(r['d_cf_audio'] + r['d_cf_video']), '-d_cf_sum full720')
    for nm, fn in [('margin_full', lambda r: r['margin_full']),
                   ('-margin_noctx', lambda r: -r['margin_noctx'])]:
        aucs[f'phd_q2.5_full_{nm}'] = auc_block(phd_full, fn, nm)
    # pubmedqa: probe field check
    pmq_sig = 'd_cf' if 'd_cf' in pmq_full[0] else 'delta_passage'
    # probe convention delta_passage = full - nopassage -> operator signal = -delta_passage
    if 'd_cf' in pmq_full[0]:
        aucs['pubmedqa_full'] = auc_block(pmq_full, lambda r: -r['d_cf'], '-d_cf(passage)')
    else:
        aucs['pubmedqa_full'] = auc_block(pmq_full, lambda r: -r['delta_passage'], '-delta_passage')
    aucs['phd_qwen3_dev'] = auc_block(phd_q3, lambda r: r['lp_yes_noctx'] - r['lp_yes_full'], 'd_cf(noctx) dev100 qwen3')
    # cmm qwen3 probe if it has landed
    for cand in ['outputs/cf_probe/cmm_probe_qwen3.jsonl']:
        if os.path.exists(cand):
            r3 = load_jsonl(cand)
            if len(r3) >= 50:
                aucs['cmm_qwen3_probe'] = auc_block(
                    r3, lambda r: -(r.get('d_cf_audio', 0) + r.get('d_cf_video', 0)), '-d_cf_sum qwen3')
                mf_auc, _ = bootstrap_auc([r['margin_full'] for r in r3],
                                          [r['gt_yes'] for r in r3], n=500)
                aucs['cmm_qwen3_probe']['margin_full_auc'] = round(mf_auc, 3)

    # haloquest gate AUC (test split, from the gate result json)
    gate = json.load(open('outputs/answerability/answerability_gate_result.json'))
    st1 = gate['report_stage1']
    gate_auc = st1['AUC_score_test']['AUC']
    gate_ci = st1['AUC_score_test']['boot_CI95']

    res['probe_aucs'] = aucs
    res['haloquest_gate'] = dict(auc=gate_auc, ci=gate_ci, n_test=gate.get('n_test') if isinstance(gate, dict) else None)

    # ---------------- gains ----------------
    gains = {}
    gains['phd_q2.5'] = gain_block(
        'phd CF lam2.0 vs base',
        'outputs/phd_base_model/phd_base_model_full_single_scored.jsonl',
        'outputs/phd_cf_confess/cf_lam2.0_full_single_scored.jsonl',
        extra_arms={'lam1.0': 'outputs/phd_cf_confess/cf_lam1.0_full_single_scored.jsonl'},
        randctl_file='outputs/phd_cf_confess/cf_lam2.0_randctl_full_single_scored.jsonl')
    gains['cmm_q2.5'] = gain_block(
        'cmm CF lam1.0 vs base',
        'outputs/cmm_base_model/cmm_base_model_full_single_scored.jsonl',
        'outputs/cmm_cf_confess/cf_lam1.0_full_single_scored.jsonl',
        extra_arms={'lam0.5': 'outputs/cmm_cf_confess/cf_lam0.5_full_single_scored.jsonl'},
        randctl_file='outputs/cmm_cf_confess/cf_lam1.0_randctl_full_single_scored.jsonl')
    gains['pubmedqa'] = gain_block(
        'pubmedqa CF lam0.25 vs base',
        'outputs/pubmedqa_base_model/pubmedqa_base_model_full_single_scored.jsonl',
        'outputs/pubmedqa_cf_confess/cf_lam0.25_full_single_scored.jsonl',
        extra_arms={'cad_v1': 'outputs/pubmedqa_base_cad_v1/cad_v1_full_single_scored.jsonl'})
    gains['phd_qwen3'] = gain_block(
        'phd CF lam4.0 vs base (Qwen3-Omni)',
        '/ultrafast_share/tsq/qwen3omni_outputs/phd_qwen3omni/phd_qwen3omni_full_single_scored.jsonl',
        'outputs/phd_cf_confess_qwen3/cf_qwen3_lam4.0_full_single_scored.jsonl',
        randctl_file='outputs/phd_cf_confess_qwen3/cf_qwen3_lam4.0_randctl_full_single_scored.jsonl')

    # haloquest gate gain: use the gate result json (same convention as published)
    m = gate['main']
    rc = gate['randctrl']
    halo = dict(n=m['overall']['n'],
                tf1_base=m['overall']['base'], tf1_gate=m['overall']['arm'],
                gain=m['overall']['delta_pp'],
                p_wilcoxon=m['overall']['wilcoxon_p'],
                needs_gate_gain=m['needs_gate(unanswerable affirmed)']['delta_pp'],
                needs_gate_p=m['needs_gate(unanswerable affirmed)']['wilcoxon_p'],
                randctrl_gain=rc['overall']['delta_pp'],
                net_gain=round(m['overall']['delta_pp'] - rc['overall']['delta_pp'], 2),
                p_vs_randctrl=gate['main_vs_rand_p'])
    gains['haloquest_gate'] = halo
    res['gains'] = gains

    # ---------------- dose-response fit ----------------
    # pre-registered points (plan): AUC -> gain
    pts = [
        ('phd@Q2.5', aucs['phd_q2.5_dev']['auc'], gains['phd_q2.5']['gain'], gains['phd_q2.5']['p_mcnemar_vs_base']),
        ('cmm@Q2.5', aucs['cmm_q2.5_full']['auc'], gains['cmm_q2.5']['gain'], gains['cmm_q2.5']['p_mcnemar_vs_base']),
        ('pubmedqa', aucs['pubmedqa_full']['auc'], gains['pubmedqa']['gain'], gains['pubmedqa']['p_mcnemar_vs_base']),
        ('phd@Qwen3', aucs['phd_qwen3_dev']['auc'], gains['phd_qwen3']['gain'], gains['phd_qwen3']['p_mcnemar_vs_base']),
        ('haloquest_gate', gate_auc, halo['gain'], halo['p_wilcoxon']),
    ]
    if 'cmm_qwen3_probe' in aucs:
        # verdict from EXPERIMENT_DESIGN.md cmm@Qwen3 probe section: pre-
        # registered conjunctive criterion FAILED (polarity reversed); offline
        # lambda grid inert on dev98 (base F1 82.14, any lambda >= 0.25 does
        # not improve); full e2e NOT run per pre-registration.
        pts.append(('cmm@Qwen3(polarity-reversed, offline-inert)', aucs['cmm_qwen3_probe']['auc'], 0.0, None))

    x = np.array([p[1] for p in pts if p[1] is not None and p[2] is not None], dtype=float)
    y = np.array([p[2] for p in pts if p[1] is not None and p[2] is not None], dtype=float)
    fit = {}
    if len(x) >= 3:
        lr = stats.linregress(x, y)
        fit['linear'] = dict(slope=round(lr.slope, 2), intercept=round(lr.intercept, 2),
                             r2=round(lr.rvalue ** 2, 3), p=round(lr.pvalue, 4))
        fit['spearman'] = dict(rho=round(stats.spearmanr(x, y).statistic, 3),
                               p=round(stats.spearmanr(x, y).pvalue, 4))
        # threshold model: breakpoint 0.6 (plan's expectation), slopes below/above
        below = x < 0.6
        fit['threshold_0.6'] = dict(
            n_below=int(below.sum()), mean_gain_below=round(float(y[below].mean()), 2) if below.any() else None,
            n_above=int((~below).sum()), mean_gain_above=round(float(y[~below].mean()), 2) if (~below).any() else None)
        # same-structure subset: exclude the polarity-reversed cmm@Qwen3 point
        names = [p[0] for p in pts if p[1] is not None and p[2] is not None]
        keep = [i for i, nm in enumerate(names) if 'polarity' not in nm]
        xs2 = x[keep]; ys2 = y[keep]
        lr2 = stats.linregress(xs2, ys2)
        fit['same_structure_subset'] = dict(
            excluded='cmm@Qwen3 (marginal AUC 0.721 but polarity reversed -> '
                     'offline-inert, pre-registered kill)',
            n=len(keep), slope=round(lr2.slope, 2), intercept=round(lr2.intercept, 2),
            r2=round(lr2.rvalue ** 2, 3), p=round(lr2.pvalue, 4))
        # net-of-random-control gains (kills the prior-shift artifact)
        net = []
        for lbl, g in [('phd@Q2.5', gains['phd_q2.5']), ('cmm@Q2.5', gains['cmm_q2.5']),
                       ('phd@Qwen3', gains['phd_qwen3'])]:
            if 'net_gain' in g:
                net.append((lbl, g['net_gain']))
        fit['net_of_randctl'] = net
    res['points'] = [dict(name=p[0], auc=p[1], gain=p[2], p=p[3]) for p in pts]
    res['fit'] = fit

    with open(OUT, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)

    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
