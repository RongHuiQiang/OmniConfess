# -*- coding: utf-8 -*-
"""answerability_contract_gate.py — 叠加实验：契约 v2 × 可答性门（两张赢牌合体）.

Coordinator brief (pre-registered):
  base v2 (contract: min-length + fixed avoidance template, over-triggers
  avoidance 324/365, Token-F1 35.41) + answerability gate (precision .707,
  94 flags, v1+gate 37.65) are complementary: the contract fixes the LENGTH
  disease, the gate knows WHEN to avoid.

Arbitration design. The gate decision (fit on the v1 trajectory probe,
τ/dev protocol unchanged) routes every item; the contract output supplies
well-formed affirmative answers:
  C1 (clean two-rule policy)
    R1  v1 pred already avoidance-phrased (EVADERE) -> keep v1 pred
        (v1's own evasion phrasing outscores the v2 fixed template,
         41.7 vs 36.1 on that 190-item subset)
    R2  v1 pred affirmative:
          gate flags (barecount | score<tau) -> T3 question-echo avoidance
          gate passes -> take the v2 answer (contract-complete affirmative
                         when v2 affirmed, v2 avoidance when v2 avoided --
                         the gate-passed-but-v2-avoided items are mostly
                         gate misses where v2's template beats v1's
                         hallucinated affirmation)
  Pre-registered family selected on the SAME 100-sample dev split:
    A0 v1+gate (existing)        A1 = A0 + nonflag&v2affirm->v2
    A2 = A1 + nonflag&v2avoid&v1affirm->v2 (= C1 modulo the v1evade rule)
    A3 = A2 + flagged&v2avoid->v2 phrasing
    A4 = A2 + flagged&v2affirm->v2
  Random control: same policy structure, flag set drawn uniformly at random
  (same count, seed 49).

Judgement (full 365, Token-F1; GAVIE deferred - zhipu balance exhausted):
  combined > 37.65 (gate alone) AND > 35.41 (contract alone);
  avoidance rate in 90-140/365 (anchored at the 127 strict-invisible items).

Usage:
    python answerability_contract_gate.py
"""
import os, sys, json, random, re
from collections import Counter

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from answerability_gate import token_f1, TEMPLATES  # noqa: E402
from probe_claim_cf import EVADERE  # noqa: E402

V1_PATH = os.path.join(HERE, 'outputs', 'haloquest_base_model',
                       'haloquest_base_model_full_single_scored.jsonl')
V2_PATH = os.path.join(HERE, 'outputs', 'haloquest_contract_v2',
                       'haloquest_contract_v2_full_single_scored.jsonl')
GATE_PATH = os.path.join(HERE, 'outputs', 'answerability',
                         'haloquest_answerability_gate_full.jsonl')
PROBE_PATH = os.path.join(HERE, 'outputs', 'answerability',
                          'haloquest_answerability_probe_p3.json')

# v2 fixed-template avoidance detector
V2_AVOID = re.compile(r'there\s+is\s+no\b|not\s+visible|cannot|can\s*not', re.I)
# strict "invisible gt" (the coordinator's 127 anchor uses a stricter cut; we
# report both this and the EVADERE-neg view)
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
    from answerability_gate import stratified_split
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

    res = {'dev_grid': dev_grid, 'policy': best, 'n_flag_gate': n_flag}
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

    # judgement
    c = blocks[f'combined({best})']['overall']
    res['criteria'] = {
        'gt_37.65': bool(c > 37.65), 'gt_35.41': bool(c > 35.41),
        'avoid_band_90_140': bool(90 <= blocks[f'combined({best})']['n_avoid'] <= 140),
        'combined_overall': c, 'n_avoid': blocks[f'combined({best})']['n_avoid'],
        'n_strict_invisible_gt': len(strict_ids),
        'n_evade_neg_gt': len(neg_ids)}

    outdir = os.path.join(HERE, 'outputs', 'answerability_contract')
    os.makedirs(outdir, exist_ok=True)
    combined_path = os.path.join(outdir, 'haloquest_answerability_contract_gate_full.jsonl')
    with open(combined_path, 'w') as f:
        for r in v1.values():
            i = r['id']
            f.write(json.dumps({'id': i, 'question': r['question'], 'gt': gt[i],
                                'group_key': r.get('group_key'), 'dataset': 'haloquest',
                                'method': 'answerability_contract_gate',
                                'base_pred': r['pred'], 'v2_pred': v2[i]['pred'],
                                'flagged': meta[i]['flagged'],
                                'pred': best_fn(i)}, ensure_ascii=False) + '\n')
    rand_path = os.path.join(outdir, 'haloquest_answerability_contract_gate_randctrl.jsonl')
    with open(rand_path, 'w') as f:
        for r in v1.values():
            i = r['id']
            f.write(json.dumps({'id': i, 'question': r['question'], 'gt': gt[i],
                                'group_key': r.get('group_key'), 'dataset': 'haloquest',
                                'method': 'answerability_contract_gate_randctrl',
                                'pred': rand_C1(i)}, ensure_ascii=False) + '\n')
    json.dump(res, open(os.path.join(outdir, 'contract_gate_result.json'), 'w'),
              indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f'\n[done] -> {combined_path}\n        -> {rand_path}')


if __name__ == '__main__':
    main()
