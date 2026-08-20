#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A1 (ABLATION_PLAN.md): CF component ablation -- offline arms.

Pre-registered design (before running):
  Arms at the decision token, lambda frozen at the shipped value (phd 2.0,
  cmm 1.0), operator identical to the frozen cf_confess_*:
    1. no view (= base)                        [online, on disk]
    2. view = context removed (shipped CF)     [online, on disk]
    3. view = image removed  (noimg)           [offline sim from probe]
    4. view = question only    (double mask)   [GPU probe + offline sim]
    5. random sign                             [online, on disk]
    6. fixed-direction CAD-style degenerate arm at the decision token:
       margin' = (1+alpha)*margin_full - alpha*margin_view, view=noctx,
       alpha in {0.25,0.5,1,2,4} -- best-alpha reported (criterion: < CF)
    7. misplacement intervention               [online GPU arm, cf_misplace5]
  Criteria (plan): misplacement F1 < 80 (vs 84.60); fixed-direction < CF.
  Directional prediction (not a registered criterion): the noimg view arm
  degrades below base (probe AUC 0.752 in probe convention = anti-informative
  in operator convention).

Offline/online equivalence: validated 901/901 (phd) and 720/720 (cmm) in the
shipped experiment sections; re-validated here against the online arms before
trusting the offline-only arms.
"""
import os
import sys
import json
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import load_jsonl, judgement_preds, f1_yp, paired_mcnemar_from_preds, mcnemar_exact

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'a1_offline.json')

YES, NO = 1, 0


def sim_arm(rows, margin_fn):
    """margin_fn(row) -> adjusted margin; forced Yes iff > 0."""
    yt, yp = [], []
    for r in rows:
        m = margin_fn(r)
        if m is None or (isinstance(m, float) and m != m):
            continue
        yt.append(r['gt_yes'])
        yp.append(YES if m > 0 else NO)
    return yt, yp


def f1_of(yt, yp):
    return round(f1_score(yt, yp, zero_division=0) * 100, 2)


def main():
    res = {}
    phd = load_jsonl('outputs/cf_probe/phd_probe_full.jsonl')
    # operator signal conventions: d = lp_yes(view) - lp_yes(full)
    lam = 2.0

    # ---- validation: offline sim of the shipped arm vs online ----
    yt, yp = sim_arm(phd, lambda r: r['margin_full'] + lam * (r['lp_yes_noctx'] - r['lp_yes_full']))
    res['validate_phd_offline_vs_online'] = dict(
        offline_f1=f1_of(yt, yp), n=len(yt),
        online_f1=84.60, online_note='outputs/phd_cf_confess/cf_lam2.0')

    # ---- view ablation ----
    views = {
        'noctx (shipped CF view)': lambda r: r['margin_full'] + lam * (r['lp_yes_noctx'] - r['lp_yes_full']),
        'noimg (image removed)': lambda r: r['margin_full'] + lam * (r['lp_yes_noimg'] - r['lp_yes_full']),
    }
    qonly_path = 'outputs/cf_probe/phd_probe_qonly.jsonl'
    qonly = None
    rows_j = []
    if os.path.exists(qonly_path) and os.path.getsize(qonly_path) > 1000:
        qonly = {}
        for r in load_jsonl(qonly_path):
            # join by (id, gt, pred, dec_index) -- ids are shard-local
            qonly[(str(r['id']), str(r['gt']), str(r['pred']), r['dec_index'])] = r
        joined, miss = 0, 0
        rows_j = []
        for r in phd:
            k = (str(r['id']), str(r['gt']), str(r['pred']), r['dec_index'])
            if k in qonly:
                rows_j.append((r, qonly[k]))
                joined += 1
            else:
                miss += 1
        res['qonly_join'] = dict(joined=joined, missing=miss)
        views['qonly (double mask)'] = None  # filled below with joined rows

    tbl = {}
    for name, fn in views.items():
        if fn is None:
            continue
        yt, yp = sim_arm(phd, fn)
        tbl[name] = dict(f1=f1_of(yt, yp), n=len(yt), kind='offline sim')
    if qonly is not None:
        yt, yp = [], []
        for r, q in rows_j:
            m = r['margin_full'] + lam * (q['lp_yes_qonly'] - r['lp_yes_full'])
            yt.append(r['gt_yes'])
            yp.append(YES if m > 0 else NO)
        tbl['qonly (double mask)'] = dict(f1=f1_of(yt, yp), n=len(yt),
                                          kind='offline sim on GPU-probed view')
    # online anchors
    tbl['no view (= base)'] = dict(f1=71.68, n=906, kind='online')
    tbl['random sign'] = dict(f1=76.91, n=906, kind='online')
    res['phd_view_ablation'] = tbl

    # ---- degenerate fixed-direction (CAD-style) arm at the decision token ----
    deg = {}
    for alpha in [0.25, 0.5, 1.0, 2.0, 4.0]:
        fn = lambda r, a=alpha: (1 + a) * r['margin_full'] - a * r['margin_noctx']
        yt, yp = sim_arm(phd, fn)
        deg[f'alpha={alpha}'] = f1_of(yt, yp)
    best_alpha = max(deg, key=lambda k: deg[k])
    # second degenerate form: CF with a FIXED (negative) lambda, the literal
    # "CF == fixed-lambda CAD" identity  margin' = margin_full - alpha*d_yes
    deg2 = {}
    for alpha in [0.25, 0.5, 1.0, 2.0, 4.0]:
        yt, yp = sim_arm(phd, lambda r, a=alpha:
                         r['margin_full'] - a * (r['lp_yes_noctx'] - r['lp_yes_full']))
        deg2[f'alpha={alpha}'] = f1_of(yt, yp)
    best_alpha2 = max(deg2, key=lambda k: deg2[k])
    a2 = float(best_alpha2.split('=')[1])
    _, yp_fx = sim_arm(phd, lambda r: r['margin_full'] - a2 * (r['lp_yes_noctx'] - r['lp_yes_full']))
    agree2 = tot2 = 0
    for r in phd:
        m_cf = r['margin_full'] + lam * (r['lp_yes_noctx'] - r['lp_yes_full'])
        m_fx = r['margin_full'] - a2 * (r['lp_yes_noctx'] - r['lp_yes_full'])
        agree2 += int((m_cf > 0) == (m_fx > 0))
        tot2 += 1
    res['phd_degenerate_fixed_direction'] = dict(
        grid_cadlogit=deg, best_cadlogit=best_alpha, best_f1_cadlogit=deg[best_alpha],
        grid_fixedlambda=deg2, best_fixedlambda=best_alpha2, best_f1_fixedlambda=deg2[best_alpha2],
        formula_cadlogit="margin' = (1+alpha)*margin_full - alpha*margin_noctx",
        formula_fixedlambda="margin' = margin_full - alpha*d_yes  (CF with fixed negative lambda)",
        first_token_agreement_fixedlambda_vs_CF=dict(agree=agree2, tot=tot2,
                                                     rate=round(agree2 / tot2 * 100, 2)),
        criterion='best-alpha F1 < CF 84.60 (both forms)')
    # first-token agreement between the degenerate best arm and the shipped CF arm
    a = float(best_alpha.split('=')[1])
    agree = tot = 0
    for r in phd:
        m_cf = r['margin_full'] + lam * (r['lp_yes_noctx'] - r['lp_yes_full'])
        m_deg = (1 + a) * r['margin_full'] - a * r['margin_noctx']
        if m_cf != m_cf or m_deg != m_deg:
            continue
        agree += int((m_cf > 0) == (m_deg > 0))
        tot += 1
    res['phd_degenerate_fixed_direction']['first_token_agreement_with_CF'] = dict(
        agree=agree, tot=tot, rate=round(agree / tot * 100, 2) if tot else None)
    # and with the ACTUAL online CAD run (v0 prompt caveat) for reference
    res['phd_degenerate_fixed_direction']['online_cad_reference'] = dict(
        f1=79.40, note='v0 prompt queue arm, different prompt, reference only')

    # ---- degeneracy equivalence check on pubmedqa (same-prompt CAD online) ----
    pmq = load_jsonl('outputs/cf_probe/pubmedqa_probe_full.jsonl')
    cad = {}
    by_key = {}
    for r in load_jsonl('outputs/pubmedqa_base_cad_v1/cad_v1_full_single_scored.jsonl'):
        by_key[str(r['id'])] = r['eval_metrics']['Prediction']
    for lamf in [-0.5, -1.0, -2.0]:
        agree = tot = 0
        for r in pmq:
            m = r['margin_full'] + lamf * r['d_cf']
            k = str(r['id'])
            if k not in by_key:
                continue
            agree += int((1 if m > 0 else 0) == by_key[k])
            tot += 1
        cad[f'lambda={lamf}'] = dict(agree=agree, tot=tot,
                                     rate=round(agree / tot * 100, 2) if tot else None)
    # how much does actual CAD differ from base first tokens?
    basep = {}
    for r in load_jsonl('outputs/pubmedqa_base_model/pubmedqa_base_model_full_single_scored.jsonl'):
        basep[str(r['id'])] = r['eval_metrics']['Prediction']
    diff = sum(1 for k in by_key if k in basep and by_key[k] != basep[k])
    res['pubmedqa_cad_equivalence'] = dict(
        agreement_grid=cad,
        cad_vs_base_first_token_changes=dict(
            changed=diff, tot=len([k for k in by_key if k in basep])),
        registered_finding='offline CF at lambda in [-0.5,-2] reproduces actual '
                           'CAD first tokens 93-94% (design doc, mechanism-distinction section)')

    # ---- paired tests: shipped CF vs each offline arm ----
    def preds(fn):
        return sim_arm(phd, fn)
    yt_cf, yp_cf = preds(views['noctx (shipped CF view)'])
    tests = {}
    for nm, fn in [('noimg', views['noimg (image removed)']),
                   ('qonly', views.get('qonly (double mask)'))]:
        if fn is None and nm == 'qonly':
            yt2, yp2 = [], []
            for r, q in rows_j:
                m = r['margin_full'] + lam * (q['lp_yes_qonly'] - r['lp_yes_full'])
                yt2.append(r['gt_yes'])
                yp2.append(YES if m > 0 else NO)
        else:
            yt2, yp2 = preds(fn)
        if len(yt2) == len(yt_cf):
            b, c, p = paired_mcnemar_from_preds(yp_cf, yp2, yt_cf)
            tests[f'CF_vs_{nm}'] = dict(arm_fixes=b, arm_breaks=c, p=p)
    # degenerate best alpha vs CF
    yt3, yp3 = sim_arm(phd, lambda r: (1 + a) * r['margin_full'] - a * r['margin_noctx'])
    b, c, p = paired_mcnemar_from_preds(yp_cf, yp3, yt_cf)
    tests['CF_vs_degenerate_cadlogit'] = dict(arm_fixes=b, arm_breaks=c, p=p)
    # fixed-lambda CF form vs CF
    b, c, p = paired_mcnemar_from_preds(yp_cf, yp_fx, yt_cf)
    tests['CF_vs_degenerate_fixedlambda'] = dict(arm_fixes=b, arm_breaks=c, p=p)
    # noimg vs base (does the wrong view actively hurt?)
    yt0, yp0 = sim_arm(phd, lambda r: r['margin_full'])
    yt2, yp2 = preds(views['noimg (image removed)'])
    b, c, p = paired_mcnemar_from_preds(yp0, yp2, yt0)
    tests['base_vs_noimg'] = dict(noimg_fixes=b, noimg_breaks=c, p=p)
    res['phd_paired_tests'] = tests

    # ---------------- cmm ----------------
    cmm = load_jsonl('outputs/cf_probe/cmm_probe_full.jsonl')
    lam_c = 1.0
    yt, yp = sim_arm(cmm, lambda r: r['margin_full'] + lam_c * r['d_cf_sum'])
    res['validate_cmm_offline_vs_online'] = dict(
        offline_f1=f1_of(yt, yp), n=len(yt), online_f1=76.36)
    ctbl = {}
    for nm, fn in [
            ('both views (shipped, w=1,1)', lambda r: r['margin_full'] + lam_c * r['d_cf_sum']),
            ('audio only (w=1,0)', lambda r: r['margin_full'] + lam_c * r['d_cf_audio']),
            ('video only (w=0,1)', lambda r: r['margin_full'] + lam_c * r['d_cf_video'])]:
        yt, yp = sim_arm(cmm, fn)
        ctbl[nm] = dict(f1=f1_of(yt, yp), n=len(yt))
    ctbl['no view (= base)'] = dict(f1=70.07, n=721, kind='online')
    ctbl['random sign'] = dict(f1=67.62, n=721, kind='online')
    # double-masked (nomedia) view: join the GPU probed view by (id,gt,pred,dec)
    nm_path = 'outputs/cf_probe/cmm_probe_nomedia.jsonl'
    if os.path.exists(nm_path) and os.path.getsize(nm_path) > 1000:
        nm = {}
        for r in load_jsonl(nm_path):
            nm[(str(r['id']), str(r['gt']), str(r['pred']), r['dec_index'])] = r
        yt, yp = [], []
        joined = 0
        for r in cmm:
            k = (str(r['id']), str(r['gt']), str(r['pred']), r['dec_index'])
            q = nm.get(k)
            if q is None:
                continue
            joined += 1
            d = q['lp_yes_nomedia'] - r['lp_yes_full']
            yt.append(r['gt_yes'])
            yp.append(YES if r['margin_full'] + lam_c * d > 0 else NO)
        ctbl['double mask (nomedia, GPU probe)'] = dict(
            f1=f1_of(yt, yp), n=len(yt), joined=joined)
    # degenerate: no amateur view with both margins -> use each single view & mean
    for alpha in [0.25, 0.5, 1.0, 2.0, 4.0]:
        yt, yp = sim_arm(cmm, lambda r, a=alpha:
                         (1 + a) * r['margin_full'] - a * (r['margin_noaudio'] + r['margin_novideo']) / 2)
        ctbl.setdefault('degenerate alpha grid', {})[f'alpha={alpha}'] = f1_of(yt, yp)
    res['cmm_view_ablation'] = ctbl

    with open(OUT, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
