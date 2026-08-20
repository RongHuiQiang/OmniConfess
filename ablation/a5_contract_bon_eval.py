#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A5 evaluation: contract + BoN (no gate) cell -> completes the 2^3 table."""
import os
import sys
import json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import load_jsonl

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(HERE))
sys.path.insert(0, os.getcwd())
from answerability_gate import token_f1  # noqa: E402
from bon_grounding import rank_value  # noqa: E402

OUT = os.path.join(HERE, 'a5_contract_bon_cell.json')


def main():
    rows = load_jsonl('outputs/haloquest_a5/haloquest_a5_contract_bon.jsonl')
    base = {r['id']: r['eval_metrics']['Token-F1'] for r in load_jsonl(
        'outputs/haloquest_base_model/haloquest_base_model_full_single_scored.jsonl')}
    contract = {r['id']: r['eval_metrics']['Token-F1'] for r in load_jsonl(
        'outputs/haloquest_contract_v2/haloquest_contract_v2_full_single_scored.jsonl')}
    combo = {r['id']: r['eval_metrics']['Token-F1'] for r in load_jsonl(
        'outputs/answerability_contract/haloquest_answerability_contract_gate_full_single_scored.jsonl')}

    f1s, f1s_rand, f1s_base, f1s_c, f1s_combo = [], [], [], [], []
    src = {'reused_stacked_pool': 0, 'generated': 0}
    for r in rows:
        f = token_f1(r['pred'], r['gt']) * 100
        f1s.append(f)
        src[r.get('source')] = src.get(r.get('source'), 0) + 1
        tr = r.get('trajectories') or []
        if tr and r.get('random_sel_rollout') is not None:
            ri = min(r['random_sel_rollout'], len(tr) - 1)
            f1s_rand.append(token_f1(tr[ri]['pred'], r['gt']) * 100)
        else:
            f1s_rand.append(f)
        i = r['id']
        if i in base:
            f1s_base.append(base[i])
        if i in contract:
            f1s_c.append(contract[i])
        if i in combo:
            f1s_combo.append(combo[i])

    m = lambda v: round(float(np.mean(v)), 2)
    res = dict(
        cell='contract + BoN (no gate)', n=len(rows), sources=src,
        tf1_cell=m(f1s), tf1_random_sel=m(f1s_rand),
        tf1_base=m(f1s_base), tf1_contract_only=m(f1s_c), tf1_contract_gate=m(f1s_combo),
        d_vs_base=round(m(f1s) - m(f1s_base), 2),
        p_vs_base=float(stats.wilcoxon([a - b for a, b in zip(f1s, f1s_base)]).pvalue),
        d_vs_contract_only=round(m(f1s) - m(f1s_c), 2),
        p_vs_contract_only=float(stats.wilcoxon([a - b for a, b in zip(f1s, f1s_c)]).pvalue),
        d_vs_random_sel=round(m(f1s) - m(f1s_rand), 2),
        p_vs_random_sel=float(stats.wilcoxon([a - b for a, b in zip(f1s, f1s_rand)]).pvalue),
        d_vs_contract_gate=round(m(f1s) - m(f1s_combo), 2),
        p_vs_contract_gate=float(stats.wilcoxon([a - b for a, b in zip(f1s, f1s_combo)]).pvalue))
    with open(OUT, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
