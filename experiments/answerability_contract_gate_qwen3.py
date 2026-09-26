# -*- coding: utf-8 -*-
"""answerability_contract_gate_qwen3.py — Qwen3-Omni port of the contract x gate
combine (阶段2：契约v2 × 可答性门合体).

Reproduces the Qwen2.5 A0-A4/C1 route family (answerability_contract_gate.py)
on Qwen3-Omni. The gate decision routes every item; the contract (v2) supplies
well-formed affirmative answers. Best family member selected on the SAME 100-
sample dev split as the gate, then evaluated on full 365 (Token-F1, GAVIE frozen).

Logic byte-identical to answerability_contract_gate.py; only the 4 input paths
and the output dir are the Qwen3 arm:
  V1   = Qwen3 base_model haloquest trajectory (Token-F1 33.38, RESULTS line 70)
  V2   = Qwen3 contract v2 (Token-F1 37.32, RESULTS line 110)
  GATE = Qwen3 gate operator output (answerability_gate_qwen3.py)
  PROBE= Qwen3 answerability probe json (answerability_probe_qwen3.py)

Pre-registered criterion (coordinator): combined > 37.32 (Qwen3 contract alone)
AND significant -> gate super-additivity transfers cross-backbone (对标 Q2.5
35.41 -> 39.87 super-additive point).

Usage:
    python answerability_contract_gate_qwen3.py
"""
import os, sys, json, random, re

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from answerability_gate import token_f1, TEMPLATES, stratified_split  # noqa: E402
from probe_claim_cf import EVADERE  # noqa: E402

V1_PATH = ('/ultrafast_share/tsq/qwen3omni_outputs/haloquest_qwen3omni/'
           'haloquest_qwen3omni_full_single_scored.jsonl')
V2_PATH = os.path.join(HERE, 'outputs', 'haloquest_qwen3omni_contractv2',
                       'haloquest_qwen3omni_contractv2_full_single_scored.jsonl')
GATE_PATH = os.path.join(HERE, 'outputs', 'answerability_qwen3',
                         'haloquest_answerability_gate_qwen3_full.jsonl')
PROBE_PATH = os.path.join(HERE, 'outputs', 'answerability_qwen3',
                          'haloquest_answerability_probe_qwen3_p3.json')

# v2 fixed-template avoidance detector (same as Q2.5)
V2_AVOID = re.compile(r'there\s+is\s+no\b|not\s+visible|cannot|can\s*not', re.I)
STRICT_INVISIBLE = re.compile(r'not\s+visible|there\s+(?:is|are)\s+no', re.I)


def is_avoid_text(t):
    return bool(V2_AVOID.search(t or '') or EVADERE.search(t or ''))


def main():
    v1 = {r['id']: r for r in map(json.loads, open(V1_PATH))}
    v2 = {r['id']: r for r in map(json.loads, open(V2_PATH))}
    gate = {r['id']: r for r in map(json.loads, open(GATE_PATH))}
    probe_rows = [r for r in json.load(open(PROBE_PATH))['rows'] if r['ok']]

    ids = [r['id'] for r in v1.values()]
    meta = {}
    for i in ids:
        meta[i] = {
            'v1_evade': bool(EVADERE.search(v1[i]['pred'] or '')),
            'v2_avoid': bool(V2_AVOID.search(v2[i]['pred'] or '')),
            'flagged': bool(gate[i]['flagged']),
        }
    gt = {i: v1[i]['gt'] for i in ids}
    T3 = TEMPLATES['T3']

    def pred_A0(i):
        return gate[i]['pred']

    def pred_A1(i):
        m = meta[i]
        if not m['flagged'] and not m['v2_avoid']:
            return v2[i]['pred']
        return gate[i]['pred']

    def pred_A2(i):
        m = meta[i]
        if not m['flagged'] and not m['v1_evade']:
            return v2[i]['pred']
        return gate[i]['pred']

    def pred_A3(i):
        m = meta[i]
        if m['flagged'] and m['v2_avoid']:
            return v2[i]['pred']
        return pred_A2(i)

    def pred_A4(i):
        m = meta[i]
        if m['flagged'] and not m['v2_avoid']:
            return v2[i]['pred']
        return pred_A2(i)

    def pred_C1(i):
        m = meta[i]
        if m['v1_evade']:
            return v1[i]['pred']
        if m['flagged']:
            return T3(v1[i]['question'])
        return v2[i]['pred']

    # ---- dev/test split: identical protocol to the gate (seed 42, 100 dev) --
    row_by_id = {r['id']: r for r in probe_rows}
    dev, test = stratified_split(probe_rows, 100, 42)
    dev_ids = {r['id'] for r in dev}
    test_ids = {r['id'] for r in test}

    def F(fn, idset):
        return float(np.mean([token_f1(fn(i), gt[i]) for i in idset]))

    fam = {'A0': pred_A0, 'A1': pred_A1, 'A2': pred_A2, 'A3': pred_A3,
           'A4': pred_A4, 'C1': pred_C1}
    dev_grid = {k: round(F(f, [i for i in dev_ids if i in meta]) * 100, 3)
                for k, f in fam.items()}
    best = max(dev_grid, key=dev_grid.get)
    print('[dev grid (100, same split as the gate)]', dev_grid, '->', best)
    best_fn = fam[best]

    # ---- random-flag control with the same policy structure --------------
    rng = random.Random(49)
    n_flag = sum(1 for i in ids if meta[i]['flagged'])
    eligible = [i for i in ids if not meta[i]['v1_evade'] and i in row_by_id]
    rand_flag = set(rng.sample(eligible, min(n_flag, len(eligible))))

    def rand_C1(i):
        m = meta[i]
        if m['v1_evade']:
            return v1[i]['pred']
        if i in rand_flag:
            return T3(v1[i]['question'])
        return v2[i]['pred']

    # ---- evaluate ----------------------------------------------------------
    neg_ids = {i for i in ids if EVADERE.search(gt[i] or '')}
    pos_ids = set(ids) - neg_ids
    strict_ids = {i for i in ids if STRICT_INVISIBLE.search(gt[i] or '')}

    def block(fn, name):
        preds = {i: fn(i) for i in ids}
        f1 = {i: token_f1(preds[i], gt[i]) for i in ids}
        out = {'arm': name,
               'overall': round(100 * float(np.mean([f1[i] for i in ids])), 2),
               'needs_gate': round(100 * float(np.mean(
                   [f1[i] for i in ids if row_by_id.get(i, {}).get('needs_gate')])), 2),
               'pos_gt': round(100 * float(np.mean([f1[i] for i in pos_ids])), 2),
               'test_only': round(100 * float(np.mean(
                   [f1[i] for i in ids if i in test_ids])), 2),
               'avoid_rate': round(float(np.mean(
                   [1 if is_avoid_text(preds[i]) else 0 for i in ids])), 3),
               'n_avoid': int(sum(1 if is_avoid_text(preds[i]) else 0 for i in ids))}
        return out, f1, preds

    res = {'dev_grid': dev_grid, 'policy': best, 'n_flag_gate': n_flag,
           'backbone': 'Qwen3-Omni-30B-A3B-Instruct'}
    blocks = {}
    f1s = {}
    for name, fn in [('v1_base', lambda i: v1[i]['pred']),
                     ('v2_contract', lambda i: v2[i]['pred']),
                     ('v1_gate(A0)', pred_A0),
                     (f'combined({best})', best_fn),
                     ('combined_randflag', rand_C1)]:
        blocks[name], f1s[name], _ = block(fn, name)
    res['arms'] = blocks

    # paired stats: combined vs both singles
    for other in ('v1_gate(A0)', 'v2_contract', 'v1_base'):
        a = np.array([f1s[f'combined({best})'][i] for i in ids])
        b = np.array([f1s[other][i] for i in ids])
        dd = a - b
        p = wilcoxon(a, b).pvalue if np.any(dd != 0) else 1.0
        res[f'paired_vs_{other}'] = {
            'delta_pp': round(float(dd.mean()) * 100, 2),
            'wins/ties/losses': [int((dd > 0).sum()), int((dd == 0).sum()),
                                 int((dd < 0).sum())],
            'wilcoxon_p': round(float(p), 6)}
    a = np.array([f1s[f'combined({best})'][i] for i in ids])
    b = np.array([f1s['combined_randflag'][i] for i in ids])
    res['paired_vs_randflag'] = {'delta_pp': round(float((a - b).mean()) * 100, 2),
                                 'wilcoxon_p': round(float(wilcoxon(a, b).pvalue), 6)}

    # ---- super-additivity decomposition (对标 Q2.5 35.41->39.87 超加点) ----
    # additive expectation = base + (contract-base) + (gate-base)
    base_o = blocks['v1_base']['overall']
    contract_o = blocks['v2_contract']['overall']
    gate_o = blocks['v1_gate(A0)']['overall']
    combined_o = blocks[f'combined({best})']['overall']
    add_expect = base_o + (contract_o - base_o) + (gate_o - base_o)
    res['superadditivity'] = {
        'base': base_o, 'contract_alone': contract_o, 'gate_alone': gate_o,
        'combined': combined_o, 'additive_expectation': round(add_expect, 2),
        'superadditive_gap': round(combined_o - add_expect, 2)}

    # judgement (Qwen3 criterion: combined > 37.32 contract-alone AND significant)
    c = combined_o
    res['criteria'] = {
        'gt_contract_37.32': bool(c > 37.32), 'gt_base_33.38': bool(c > 33.38),
        'gt_gate_alone': bool(c > gate_o),
        'sig_vs_contract_p': res['paired_vs_v2_contract']['wilcoxon_p'],
        'avoid_band_90_140': bool(90 <= blocks[f'combined({best})']['n_avoid'] <= 140),
        'combined_overall': c, 'n_avoid': blocks[f'combined({best})']['n_avoid'],
        'n_strict_invisible_gt': len(strict_ids),
        'n_evade_neg_gt': len(neg_ids)}

    outdir = os.path.join(HERE, 'outputs', 'answerability_contract_qwen3')
    os.makedirs(outdir, exist_ok=True)
    combined_path = os.path.join(outdir, 'haloquest_answerability_contract_gate_qwen3_full.jsonl')
    with open(combined_path, 'w') as f:
        for r in v1.values():
            i = r['id']
            f.write(json.dumps({'id': i, 'question': r['question'], 'gt': gt[i],
                                'group_key': r.get('group_key'), 'dataset': 'haloquest',
                                'method': 'answerability_contract_gate_qwen3',
                                'base_pred': r['pred'], 'v2_pred': v2[i]['pred'],
                                'flagged': meta[i]['flagged'],
                                'pred': best_fn(i)}, ensure_ascii=False) + '\n')
    rand_path = os.path.join(outdir, 'haloquest_answerability_contract_gate_qwen3_randctrl.jsonl')
    with open(rand_path, 'w') as f:
        for r in v1.values():
            i = r['id']
            f.write(json.dumps({'id': i, 'question': r['question'], 'gt': gt[i],
                                'group_key': r.get('group_key'), 'dataset': 'haloquest',
                                'method': 'answerability_contract_gate_qwen3_randctrl',
                                'pred': rand_C1(i)}, ensure_ascii=False) + '\n')
    json.dump(res, open(os.path.join(outdir, 'contract_gate_qwen3_result.json'), 'w'),
              indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f'\n[done] -> {combined_path}\n        -> {rand_path}')


if __name__ == '__main__':
    main()
