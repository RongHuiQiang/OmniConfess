#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A2 (ABLATION_PLAN.md): cost / efficiency decomposition table.

Components:
  1. per-method per-sample latency (median & mean time_sec from the run
     artifacts; wall-clock on A800, contention varies -> medians reported)
  2. CF decomposition: decision-view probe forward = CF - base per-sample
     delta (correction itself is a hard logit force ~0; intervention rate
     reported), per dataset
  3. haloquest 4-layer increment table (contract 0 extra forwards; gate =
     probe wall time from probe log; BoN = 788 s run log; feat-combo)
  4. Vote-SC: from-scratch cost = sum of arm generations; on-disk rebuild ~0
  5. efficiency-frontier data table (score vs latency per method per dataset;
     plotting left to the authors)

Zero GPU (all numbers from disk).
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import load_jsonl, judgement_preds, f1_yp

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'a2_cost.json')


def tstat(path):
    ts = [r.get('time_sec') for r in load_jsonl(path) if r.get('time_sec')]
    if not ts:
        return None
    return dict(n=len(ts), mean=round(float(np.mean(ts)), 2),
                median=round(float(np.median(ts)), 2),
                gpu_h_total=round(float(np.mean(ts)) * len(ts) / 3600, 2))


def jf1(path):
    pr = judgement_preds(load_jsonl(path))
    return round(f1_yp([x[0] for x in pr], [x[1] for x in pr]), 2)


def tf1(path):
    v = [r['eval_metrics']['Token-F1'] for r in load_jsonl(path)
         if 'eval_metrics' in r]
    return round(float(np.mean(v)), 2) if v else None


def main():
    res = {}

    # ---------------- 1+5: per-method latency & score ----------------
    tables = {}

    phd = {
        'base_model': 'outputs/phd_base_model/phd_base_model_full_single_scored.jsonl',
        'base_dola': 'outputs/phd_base_dola/phd_base_dola_full_single_scored.jsonl',
        'base_vcd': 'outputs/phd_base_vcd/phd_base_vcd_full_single_scored.jsonl',
        'base_sc': 'outputs/phd_base_sc/phd_base_sc_full_single_scored.jsonl',
        'base_cad': 'outputs/phd_base_cad/phd_base_cad_full_single_scored.jsonl',
        'base_gd': 'outputs/phd_base_gd/phd_base_gd_full_single_scored.jsonl',
        'base_pd': 'outputs/phd_base_pd/phd_base_pd_full_single_scored.jsonl',
        'base_tot': 'outputs/phd_base_tot/phd_base_tot_full_single_scored.jsonl',
        'base_opera': 'outputs/phd_base_opera/phd_base_opera_full_single_scored.jsonl',
        'base_avcd': 'outputs/phd_base_avcd/phd_base_avcd_full_single_scored.jsonl',
        'CF_lam2.0': 'outputs/phd_cf_confess/cf_lam2.0_full_single_scored.jsonl',
    }
    t = {}
    for k, p in phd.items():
        if os.path.exists(p):
            t[k] = dict(score_f1=jf1(p), **(tstat(p) or {}))
    tables['phd(n=906)'] = t

    cmm = {
        'base_model': 'outputs/cmm_base_model/cmm_base_model_full_single_scored.jsonl',
        'base_dola': 'outputs/cmm_base_dola/cmm_base_dola_full_single_scored.jsonl',
        'base_vcd': 'outputs/cmm_base_vcd/cmm_base_vcd_full_single_scored.jsonl',
        'base_sc': 'outputs/cmm_base_sc/cmm_base_sc_full_single_scored.jsonl',
        'base_cad': 'outputs/cmm_base_cad/cmm_base_cad_full_single_scored.jsonl',
        'base_avcd': 'outputs/cmm_base_avcd/cmm_base_avcd_full_single_scored.jsonl',
        'CF_lam1.0': 'outputs/cmm_cf_confess/cf_lam1.0_full_single_scored.jsonl',
    }
    t = {}
    for k, p in cmm.items():
        if os.path.exists(p):
            t[k] = dict(score_f1=jf1(p), **(tstat(p) or {}))
    tables['cmm(n=721)'] = t

    pubmedqa = {
        'base_model': 'outputs/pubmedqa_base_model/pubmedqa_base_model_full_single_scored.jsonl',
        'base_cad_v1': 'outputs/pubmedqa_base_cad_v1/cad_v1_full_single_scored.jsonl',
        'CF_lam0.25': 'outputs/pubmedqa_cf_confess/cf_lam0.25_full_single_scored.jsonl',
    }
    t = {}
    for k, p in pubmedqa.items():
        t[k] = dict(score_f1=jf1(p), **(tstat(p) or {}))
    tables['pubmedqa(n=468)'] = t

    ragtruth = {
        'base_model': 'outputs/ragtruth_base_model/ragtruth_base_model_full_single_scored.jsonl',
        'base_dola': 'outputs/ragtruth_base_dola/ragtruth_base_dola_full_single_scored.jsonl',
        'base_cad': 'outputs/ragtruth_base_cad/ragtruth_base_cad_full_single_scored.jsonl',
        'base_sc': 'outputs/ragtruth_base_sc/ragtruth_base_sc_full_single_scored.jsonl',
        'contract_v2': 'outputs/ragtruth_contract_v2/ragtruth_contract_v2_full_single_scored.jsonl',
        'contract_v3': 'outputs/ragtruth_contract_v3/ragtruth_contract_v3_full_single_scored.jsonl',
        'contract_v4': 'outputs/ragtruth_contract_v4/ragtruth_contract_v4_full_single_scored.jsonl',
    }
    t = {}
    for k, p in ragtruth.items():
        t[k] = dict(score_tf1=tf1(p), **(tstat(p) or {}))
    tables['ragtruth(n=435)'] = t

    avh = {
        'base_model(224)': 'outputs/avhbench_base_model/avhbench_base_model_full_single_scored.jsonl',
        'base_dola': 'outputs/avhbench_base_dola/avhbench_base_dola_full_single_scored.jsonl',
        'base_cad': 'outputs/avhbench_base_cad/avhbench_base_cad_full_single_scored.jsonl',
        'base_sc': 'outputs/avhbench_base_sc/avhbench_base_sc_full_single_scored.jsonl',
        'base_vcd': 'outputs/avhbench_base_vcd/avhbench_base_vcd_full_single_scored.jsonl',
        'base_avcd': 'outputs/avhbench_base_avcd/avhbench_base_avcd_full_single_scored.jsonl',
        'lookagain_res384': 'outputs/avhbench_res384/avhbench_res384_full_single_scored.jsonl',
        'lookagain_res448': 'outputs/avhbench_res448/avhbench_res448_full_single_scored.jsonl',
        'lookagain_frames16': 'outputs/avhbench_frames16/avhbench_frames16_full_single_scored.jsonl',
    }
    t = {}
    for k, p in avh.items():
        t[k] = dict(score_tf1=tf1(p), **(tstat(p) or {}))
    # vote-sc: no per-sample time (offline rebuild); from-scratch cost = arms
    arm_cost = 0.0
    for p in ['outputs/avhbench_base_model/avhbench_base_model_full_single_scored.jsonl',
              'outputs/avhbench_res384/avhbench_res384_full_single_scored.jsonl',
              'outputs/avhbench_res448/avhbench_res448_full_single_scored.jsonl',
              'outputs/avhbench_frames16/avhbench_frames16_full_single_scored.jsonl',
              'outputs/avhbench_vote_sc/temp07_arm1_full.jsonl',
              'outputs/avhbench_vote_sc/temp07_arm2_full.jsonl']:
        s = tstat(p)
        if s:
            arm_cost += s['mean'] * s['n']
    t['vote_sc_5arm(from scratch)'] = dict(
        score_tf1=tf1('outputs/avhbench_vote_sc/avhbench_vote_sc_full_single_scored.jsonl'),
        note='5 generations/sample (4 input arms + q3); q3 ~8.7 s/sample',
        gpu_h_from_scratch=round(arm_cost / 3600 + 8.7 * 645 / 3600, 2),
        rebuild_cost='~0 (offline word vote over arm outputs)')
    tables['avhbench(n=645)'] = t

    res['frontier'] = tables

    # ---------------- 2: CF decomposition ----------------
    cf = {}
    for ds, base_p, cf_p, n_views in [
            ('phd', phd['base_model'], phd['CF_lam2.0'], 1),
            ('cmm', cmm['base_model'], cmm['CF_lam1.0'], 2),
            ('pubmedqa', pubmedqa['base_model'], pubmedqa['CF_lam0.25'], 1)]:
        b, a = tstat(base_p), tstat(cf_p)
        rows = load_jsonl(cf_p)
        ir = sum(1 for r in rows if r.get('cf', {}).get('intervened')) / len(rows)
        cf[ds] = dict(
            base_mean_s=b['mean'], cf_mean_s=a['mean'],
            overhead_pct_mean=round((a['mean'] / b['mean'] - 1) * 100, 1),
            base_median_s=b['median'], cf_median_s=a['median'],
            overhead_pct_median=round((a['median'] / b['median'] - 1) * 100, 1),
            probe_forward_s_mean=round(a['mean'] - b['mean'], 2),
            probe_views=n_views,
            per_view_s_mean=round((a['mean'] - b['mean']) / n_views, 2),
            correction_cost='~0 (hard logit force at 1 token)',
            intervention_rate=round(ir * 100, 1),
            caveat='wall-clock on shared A800s; contention varies between runs')
    res['cf_decomposition'] = cf

    # ---------------- 3: haloquest 4 layers ----------------
    hq = {}
    for name, p in [('base', 'outputs/haloquest_base_model/haloquest_base_model_full_single_scored.jsonl'),
                    ('contract_v2', 'outputs/haloquest_contract_v2/haloquest_contract_v2_full_single_scored.jsonl')]:
        s = tstat(p)
        hq[name] = dict(score_tf1=tf1(p), median_s=s['median'], gpu_h_total=s['gpu_h_total'],
                        extra_forwards=0 if 'contract' in name else None)
    hq['gate_probe'] = dict(
        wall_s_per_item=0.79,
        source='outputs/answerability/probe_full.log [350/365] 277s',
        forwards_per_item='1 full-image + 3 swapped-image teacher-forced (probe design)',
        zero_extra_at_inference='routing only (flag -> echo template), no extra forward at answer time')
    hq['bon_stacked'] = dict(
        wall_s=788, n=365, s_per_item=round(788 / 365, 2),
        source='outputs/haloquest_bon_stacked_run.log "[done] 788s"',
        scope='3 candidates for the 81-item release branch (1 greedy + 3xT0.7 -> selection)')
    hq['feat_combo'] = dict(
        s_per_item='~1 forward + attention read (probe_bon_features.py on the candidate pool)',
        score_note='+0.08 over pure-delta selection (already registered)')
    res['haloquest_layers'] = hq

    with open(OUT, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
