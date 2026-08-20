# -*- coding: utf-8 -*-
"""halo_prompt_dev_eval.py — PROMPT_PROGRAM dev judgement for HaloQuest prompt
versions (offline, no GPU).

Pre-registered decision rule (coordinator brief):
  PASS if  dev_delta >= +1.0 Token-F1 points
       OR (dev_delta >= +0.5 AND paired Wilcoxon p < 0.05)
  where dev_delta = (four-layer Token-F1 of version X on the fixed 91-item dev)
                  - (published four-layer baseline on the same 91 items = 40.76)
  Small-but-significant gains are accepted per brief ("小增益也收"); anything
  else fails the round.

Usage:
    python halo_prompt_dev_eval.py v3 [v2rerun ...]
"""
import json, os, sys

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from answerability_gate import token_f1  # noqa: E402

BASE = os.path.join(HERE, 'outputs', 'haloquest_bon_stacked',
                    'haloquest_bon_stacked_feat_combo.jsonl')
DEV_IDS = json.load(open(os.path.join(
    HERE, 'OmniHalluBench', 'haloquest', 'dev_quarter', 'dev_ids.json')))


def load_preds(path):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[r['id']] = r
    missing = [i for i in DEV_IDS if i not in rows]
    assert not missing, f'missing dev ids in {path}: {missing[:5]}'
    return rows


def f1_map(rows):
    return {i: token_f1(rows[i]['pred'], rows[i]['gt']) for i in DEV_IDS}


def report(name, rows, base_rows, base_f1):
    f1 = f1_map(rows)
    d = np.array([f1[i] for i in DEV_IDS])
    b = np.array([base_f1[i] for i in DEV_IDS])
    diff = d - b
    p = wilcoxon(d, b).pvalue if np.any(diff != 0) else 1.0
    out = {
        'arm': name,
        'dev_f1': round(100 * float(d.mean()), 2),
        'baseline_dev_f1': round(100 * float(b.mean()), 2),
        'delta_pp': round(100 * float(diff.mean()), 2),
        'wins/ties/losses': [int((diff > 0).sum()), int((diff == 0).sum()),
                             int((diff < 0).sum())],
        'wilcoxon_p': round(float(p), 6),
        'pass_1.0': bool(diff.mean() * 100 >= 1.0),
        'pass_0.5_sig': bool(diff.mean() * 100 >= 0.5 and p < 0.05),
    }
    for br in ('v1evade_keep', 'flagged_T3', 'passthrough_bon'):
        ids = [i for i in DEV_IDS if rows[i]['branch'].startswith(br)]
        if ids:
            out[f'{br}'] = {
                'n': len(ids),
                'f1': round(100 * float(np.mean([f1[i] for i in ids])), 2),
                'base_f1': round(100 * float(np.mean([base_f1[i] for i in ids])), 2),
                'delta': round(100 * float(np.mean(
                    [f1[i] - base_f1[i] for i in ids])), 2)}
    return out


def main():
    base_rows = load_preds(BASE)
    base_f1 = f1_map(base_rows)
    print(f'[baseline published four-layer on dev-91] '
          f'{100*float(np.mean(list(base_f1.values()))):.2f}')
    for tag in sys.argv[1:]:
        path = os.path.join(HERE, 'outputs', 'haloquest_prompt_iter',
                            f'haloquest_prompt_{tag}_dev.jsonl')
        rows = load_preds(path)
        res = report(tag, rows, base_rows, base_f1)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        outp = path.replace('.jsonl', '_eval.json')
        json.dump(res, open(outp, 'w'), indent=2, ensure_ascii=False)

        # cross-arm pairing vs another dev run (e.g. v2 rerun noise control)
    if len(sys.argv) > 2:
        a, b = sys.argv[1], sys.argv[2]
        ra = load_preds(os.path.join(HERE, 'outputs', 'haloquest_prompt_iter',
                                     f'haloquest_prompt_{a}_dev.jsonl'))
        rb = load_preds(os.path.join(HERE, 'outputs', 'haloquest_prompt_iter',
                                     f'haloquest_prompt_{b}_dev.jsonl'))
        fa, fb = f1_map(ra), f1_map(rb)
        d = np.array([fa[i] - fb[i] for i in DEV_IDS])
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        print(f'[paired {a} vs {b}] delta {100*float(d.mean()):.2f}pp '
              f'wins/ties/losses {(int((d>0).sum()), int((d==0).sum()), int((d<0).sum()))} '
              f'p={p:.4f}')


if __name__ == '__main__':
    main()
