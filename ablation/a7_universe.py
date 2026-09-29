#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A7 (ABLATION_PLAN.md): cross-universe robustness.

Pre-registered question: does the method ranking flip when scored on the
shipped universe (filtered sets: phd_566, haloquest_608, ...) instead of our
full sets? Procedure: take the question set of the shipped BaseModel result
file, restrict our arms to those questions, rescore with the repo scorer,
compare the ranking with the full-set ranking.

Universe overlap census (computed first, determines what is non-trivial):
  phd       our 861 uniq q, shipped 565 uniq -> 405 overlap  (NON-trivial)
  pubmedqa  our 468  -> all inside shipped 500               (trivial)
  cmm       our 130  -> all inside shipped 158               (trivial)
  ragtruth  our 143  -> all inside shipped 150               (trivial)
  haloquest our 357  -> all inside shipped 591               (trivial)
  avhbench  question is a constant template -> key on (question, gt)
So the filter is only informative on phd; the trivial datasets are reported
as census rows (our set is a subset of the shipped universe, no filtering
possible). Zero GPU.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import load_jsonl, judgement_preds, f1_yp

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'a7_universe.json')


def f1_on_subset(path, subset_q):
    rows = [r for r in load_jsonl(path) if r['question'] in subset_q]
    pr = judgement_preds(rows)
    g = [x[0] for x in pr]
    p = [x[1] for x in pr]
    return round(f1_yp(g, p), 2), len(rows)


def main():
    res = {'census': {}}
    ship_sets = {
        'phd': 'results_shipped/phd_566/BaseModel_scored.jsonl',
        'pubmedqa': 'results_shipped/pubmedqa_500/BaseModel_scored.jsonl',
        'cmm': 'results_shipped/cmm_795/BaseModel_scored.jsonl',
        'ragtruth': 'results_shipped/ragtruth_740/BaseModel.jsonl',
        'haloquest': 'results_shipped/haloquest_608/BaseModel.jsonl',
        'avhbench': 'results_shipped/avhbench_645/BaseModel.jsonl',
    }
    ours_sets = {
        'phd': 'outputs/phd_base_model/phd_base_model_full_single_scored.jsonl',
        'pubmedqa': 'outputs/pubmedqa_base_model/pubmedqa_base_model_full_single_scored.jsonl',
        'cmm': 'outputs/cmm_base_model/cmm_base_model_full_single_scored.jsonl',
        'ragtruth': 'outputs/ragtruth_base_model/ragtruth_base_model_full_single_scored.jsonl',
        'haloquest': 'outputs/haloquest_base_model/haloquest_base_model_full_single_scored.jsonl',
        'avhbench': 'outputs/avhbench_base_model/avhbench_base_model_full_single_scored.jsonl',
    }
    for k in ship_sets:
        s = set(r['question'] for r in load_jsonl(ship_sets[k]))
        o = set(r['question'] for r in load_jsonl(ours_sets[k]))
        res['census'][k] = dict(shipped_uniq_q=len(s), ours_uniq_q=len(o),
                                ours_in_shipped=len(o & s),
                                trivial=bool(o <= s))
    # ---------------- phd: the non-trivial filter ----------------
    sub = set(r['question'] for r in load_jsonl(ship_sets['phd']))
    arms = {
        'base_model': 'outputs/phd_base_model/phd_base_model_full_single_scored.jsonl',
        'base_dola': 'outputs/phd_base_dola/phd_base_dola_full_single_scored.jsonl',
        'base_sc': 'outputs/phd_base_sc/phd_base_sc_full_single_scored.jsonl',
        'base_cad(v0-prompt)': 'outputs/phd_base_cad/phd_base_cad_full_single_scored.jsonl',
        'base_vcd': 'outputs/phd_base_vcd/phd_base_vcd_full_single_scored.jsonl',
        'CF_lam1.0': 'outputs/phd_cf_confess/cf_lam1.0_full_single_scored.jsonl',
        'CF_lam2.0': 'outputs/phd_cf_confess/cf_lam2.0_full_single_scored.jsonl',
        'CF_lam2.0_randctl': 'outputs/phd_cf_confess/cf_lam2.0_randctl_full_single_scored.jsonl',
    }
    tbl = {}
    for name, p in arms.items():
        f_sub, n_sub = f1_on_subset(p, sub)
        f_all, n_all = f1_on_subset(p, set(r['question'] for r in load_jsonl(p)))
        tbl[name] = dict(f1_full=f_all, n_full=n_all, f1_shipped_subset=f_sub, n_subset=n_sub)
    res['phd'] = dict(subset_uniq_q=len(sub), arms=tbl)
    # rank comparison (exclude the v0-prompt CAD row from the same-prompt ranking)
    rankable = [k for k in tbl if 'v0' not in k]
    full_rank = sorted(rankable, key=lambda k: -tbl[k]['f1_full'])
    sub_rank = sorted(rankable, key=lambda k: -tbl[k]['f1_shipped_subset'])
    res['phd']['ranking_full'] = full_rank
    res['phd']['ranking_subset'] = sub_rank
    res['phd']['ranking_preserved'] = full_rank == sub_rank
    # tau
    from scipy.stats import kendalltau
    x = [tbl[k]['f1_full'] for k in rankable]
    y = [tbl[k]['f1_shipped_subset'] for k in rankable]
    res['phd']['kendall_tau'] = round(float(kendalltau(x, y).statistic), 3)
    # paired tests on the subset: CF vs base, CF vs randctl
    from util import paired_mcnemar_from_preds, mcnemar_exact

    def preds_subset(path):
        rows = [r for r in load_jsonl(path) if r['question'] in sub]
        pr = judgement_preds(rows)
        return [x[0] for x in pr], [x[1] for x in pr]
    g, pb = preds_subset(arms['base_model'])
    _, pa = preds_subset(arms['CF_lam2.0'])
    _, prc = preds_subset(arms['CF_lam2.0_randctl'])
    b, c, p1 = paired_mcnemar_from_preds(pb, pa, g)
    b2, c2, p2 = paired_mcnemar_from_preds(pa, prc, g)
    res['phd']['paired'] = dict(
        cf_vs_base=dict(fix=b, break_=c, p=p1),
        cf_vs_randctl=dict(fix=b2, break_=c2, p=p2))

    with open(OUT, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
