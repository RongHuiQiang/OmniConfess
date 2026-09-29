#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A6 (ABLATION_PLAN.md): Vote-SC arm ablation, formalized.

Arms on disk (all 645, same vote>=2 content-word rebuild):
  2-arm subsets (from the 10.1 search: 224+q3, 384+q3) -> recomputed here
  3-arm  224+384+q3
  4-arm  224+384+448+f16  (pure input perturbation)
  4-arm  224+384+448+q3   (noF16)
  5-arm  + q3 (shipped main, 21.56)
  5-arm thr3 (threshold ablation)
  6-arm  nof16 + two temperature arms (diversity type ablation)
  7-arm  all (saturation point)
Base arm scores: 224/384/448/f16/q3 single arms + base_sc (same-compute
self-consistency) + the two temperature arms.

Scoring: repo descriptive Token-F1 (single_eval), identical to every
published number; paired Wilcoxon vs base_model keyed by (question, gt).
Zero GPU.
"""
import os
import sys
import json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import load_jsonl

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.getcwd(), 'OmniConfess', 'eval'))
from single_eval import UnifiedEvaluator  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'a6_votes_ablation.json')
VDIR = 'outputs/avhbench_vote_sc'


def score(path):
    ev = UnifiedEvaluator()
    rows = ev._load_data(path)
    for r in rows:
        r['eval_metrics'] = {'Token-F1': ev._compute_token_f1(r.get('pred'), r.get('gt')) * 100}
    return rows


def tf1(rows):
    return float(np.mean([r['eval_metrics']['Token-F1'] for r in rows]))


def pair(base_rows, arm_rows):
    b = {}
    for r in base_rows:
        b.setdefault((r['question'], str(r['gt'])), []).append(r['eval_metrics']['Token-F1'])
    d = []
    for r in arm_rows:
        k = (r['question'], str(r['gt']))
        if k in b and b[k]:
            d.append(r['eval_metrics']['Token-F1'] - b[k].pop(0))
    return d


def wilcoxon_p(d):
    d = np.asarray(d)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(d).pvalue)


def main():
    base = score('outputs/avhbench_base_model/avhbench_base_model_full.jsonl')
    base_sc = score('outputs/avhbench_base_sc/avhbench_base_sc_full.jsonl')
    res = dict(base=tf1(base), base_sc_same_compute=tf1(base_sc))

    single_arms = {
        '224': 'outputs/avhbench_base_model/avhbench_base_model_full.jsonl',
        '384': 'outputs/avhbench_res384/avhbench_res384_full.jsonl',
        '448': 'outputs/avhbench_res448/avhbench_res448_full.jsonl',
        'f16': 'outputs/avhbench_frames16/avhbench_frames16_full.jsonl',
        'q3': '/ultrafast_share/tsq/qwen3omni_outputs/avhbench_qwen3omni/avhbench_qwen3omni_full.jsonl',
        't07a1': f'{VDIR}/temp07_arm1_full.jsonl',
        't07a2': f'{VDIR}/temp07_arm2_full.jsonl',
    }
    arm_scores = {}
    for k, p in single_arms.items():
        rows = score(p)
        arm_scores[k] = round(tf1(rows), 2)
    res['single_arm_scores'] = arm_scores

    combos = {
        '2arm_224+q3': f'{VDIR}/avhbench_vote_sc_5arm_thr2_regress.jsonl',  # placeholder, replaced below
    }
    # score every shipped rebuild file
    files = {
        '4arm_input(224+384+448+f16)_thr2': None,  # pure-input 4-arm: rebuild needed? check existing
        '4arm_noF16(224+384+448+q3)_thr2': f'{VDIR}/avhbench_vote_sc_noF16_ablation.jsonl',
        '5arm(4input+q3)_thr2': f'{VDIR}/avhbench_vote_sc_full.jsonl',
        '5arm(4input+q3)_thr3': f'{VDIR}/avhbench_vote_sc_5arm_thr3.jsonl',
        '6arm_noF16(+2temp)_thr2': f'{VDIR}/avhbench_vote_sc_6arm_nof16_thr2.jsonl',
        '6arm_noF16(+2temp)_thr3': f'{VDIR}/avhbench_vote_sc_6arm_nof16_thr3.jsonl',
        '7arm(all)_thr2': f'{VDIR}/avhbench_vote_sc_7arm_thr2.jsonl',
        '7arm(all)_thr3': f'{VDIR}/avhbench_vote_sc_7arm_thr3.jsonl',
    }
    out_tbl = {}
    for name, p in files.items():
        if p is None or not os.path.exists(p):
            continue
        rows = score(p)
        d = pair(base, rows)
        out_tbl[name] = dict(
            f1=round(tf1(rows), 2), delta_vs_base=round(float(np.mean(d)), 2),
            p_wilcoxon=wilcoxon_p(d), n=len(d),
            arms=rows[0].get('arms'), thr=rows[0].get('vote_threshold'))

    # ---- missing small combos via the repo rebuild logic (2/3/pure-4 arms) --
    sys.path.insert(0, os.getcwd())
    import avhbench_vote_sc_rebuild as RB
    arms, keys = RB.load_arms()
    combos = [
        ('2arm_224+q3_thr2', ['224', 'q3']),
        ('2arm_384+q3_thr2', ['384', 'q3']),
        ('2arm_224+384_thr2', ['224', '384']),
        ('3arm_224+384+q3_thr2', ['224', '384', 'q3']),
        ('3arm_224+384+448_thr2', ['224', '384', '448']),
        ('4arm_input(224+384+448+f16)_thr2', ['224', '384', '448', 'f16']),
    ]
    import statistics
    for name, al in combos:
        outp = f'{VDIR}/a6_{name}.jsonl'
        try:
            RB.build(arms, keys, al, 2, outp, f'a6_{name}')
        except KeyError:
            pass  # build() pairs vs the '224' arm internally; we pair below
        rows = load_jsonl(outp)
        bmap = {}
        for r in base:
            bmap.setdefault((r['question'], str(r['gt'])), []).append(
                r['eval_metrics']['Token-F1'])
        d = []
        for r in rows:
            k = (r['question'], str(r['gt']))
            if bmap.get(k):
                d.append(r['local_f1'] - bmap[k].pop(0))
        out_tbl[name] = dict(
            f1=round(float(statistics.mean([r['local_f1'] for r in rows])), 2),
            delta_vs_base=round(float(np.mean(d)), 2),
            p_wilcoxon=wilcoxon_p(d), n=len(rows), arms=al, thr=2,
            scorer='rebuild token_f1 (identical to single_eval Token-F1)')
    res['arm_table'] = out_tbl

    with open(OUT, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
