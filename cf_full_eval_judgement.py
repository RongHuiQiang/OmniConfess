#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cf_full_eval_judgement.py -- end-to-end evaluation of the cmm / pubmedqa
CF-Confess arms against base / random-sign control / CAD (matched v1 prompt).

Per-sample pairing:
  cmm      : (shard, id)  -- result ids are shard-local row indices
  pubmedqa : id           -- original globally unique ids

Scoring: eval/single_eval.py judgement mode (gt=maybe excluded, maybe-pred
counted as a miss), identical to every other number in the repo.

Usage:
  python cf_full_eval_judgement.py cmm   outputs/cmm_cf_confess outputs/cmm_base_cad_v1 ...
  python cf_full_eval_judgement.py pubmedqa outputs/pubmedqa_cf_confess outputs/pubmedqa_base_cad_v1 ...
Positional args after the dataset are (label, path) pairs is NOT the format:
pass --arm label=path ... instead.
"""
import os, sys, json, argparse, glob
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'OmniConfess', 'eval'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from single_eval import UnifiedEvaluator   # noqa: E402

NSHARD = 8


def score_file(path):
    """Run the repo scorer on a result file, return rows with eval_metrics."""
    ev = UnifiedEvaluator()
    data = ev._load_data(path)
    out_path = path.rsplit('.', 1)[0] + '_single_scored.jsonl'
    with open(out_path, 'w') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    ev.evaluate_judgment_tasks(out_path, save_single=True)
    return [json.loads(l) for l in open(out_path)]


def full_from_shards(outdir, prefix):
    parts = sorted(glob.glob(os.path.join(outdir, f'{prefix}_shard*.jsonl')),
                   key=lambda p: int(p.split('_shard')[1].split('.')[0]))
    rows = []
    for k, p in enumerate(parts):
        for line in open(p):
            r = json.loads(line)
            r['_shard'] = k
            r['_local_id'] = r['id']
            rows.append(r)
    return rows


def keyed(rows, dataset):
    out = {}
    for r in rows:
        k = (r['_shard'], r['_local_id']) if dataset == 'cmm' else r['id']
        out[k] = r
    return out


def binarize(r):
    g = r['eval_metrics']['Ground_Truth']
    p = r['eval_metrics']['Prediction']
    if g == 2:
        return None, None          # unscoreable maybe-gt
    return g, (1 - g) if p == 2 else p


def metrics(pairs):
    yt = [g for g, p in pairs]
    yp = [p for g, p in pairs]
    f1 = f1_score(yt, yp, zero_division=0)
    acc = accuracy_score(yt, yp)
    tpr = recall_score(yt, yp, pos_label=1, zero_division=0)
    tnr = recall_score(yt, yp, pos_label=0, zero_division=0)
    return dict(n=len(yt), F1=round(f1 * 100, 2), acc=round(acc * 100, 2),
                yes_rate=round(float(np.mean(yp)) * 100, 2),
                TPR=round(tpr, 3), FPR=round(1 - tnr, 3),
                Youden=round(tpr - (1 - tnr), 3))


def mcnemar(a, b, yt):
    """a, b binary pred vectors; two-sided exact-ish binomial McNemar."""
    b01 = int(sum(1 for x, y, g in zip(a, b, yt) if x != y and x == g))
    b10 = int(sum(1 for x, y, g in zip(a, b, yt) if x != y and y == g))
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    from math import comb
    p = sum(comb(n, k) for k in range(min(b01, b10) + 1)) / 2 ** n * 2
    return b01, b10, min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=['cmm', 'pubmedqa'])
    ap.add_argument('--base', required=True, help='scored base full file')
    ap.add_argument('--arm', action='append', nargs=2,
                    metavar=('LABEL', 'PATH'),
                    help="label=path to a shard prefix dir/prefix or full jsonl")
    args = ap.parse_args()

    base_rows = [json.loads(l) for l in open(args.base)]
    if args.dataset == 'cmm':          # reconstruct (shard, local_id)
        assert len(base_rows) == 721
        aligned = []
        pos = 0
        for sh in range(NSHARD):
            shard = json.load(open(f'OmniHalluBench/cmm/cmm_shard{sh}of{NSHARD}.json'))
            for j, it in enumerate(shard):
                r = dict(base_rows[pos]); pos += 1
                r['_shard'], r['_local_id'] = sh, str(j)
                assert r['question'] == it['question'], \
                    f'misaligned shard {sh} row {j}'
                aligned.append(r)
        base_rows = aligned
    else:
        for r in base_rows:
            r['_shard'], r['_local_id'] = 0, r['id']

    base_bin = {}
    for r in base_rows:
        g, p = binarize(r)
        if g is not None:
            base_bin[(r['_shard'], r['_local_id']) if args.dataset == 'cmm'
                     else r['id']] = (g, p)

    print(f'== {args.dataset} ==')
    print(f'base (n={len(base_bin)}):', metrics(list(base_bin.values())))

    arms = {}
    for label, path in args.arm:
        outdir, prefix = os.path.split(path)
        rows = full_from_shards(outdir, prefix)
        rows = score_file_rows(rows)
        d = {}
        for r in rows:
            g, p = binarize(r)
            if g is None:
                continue
            k = (r['_shard'], r['_local_id']) if args.dataset == 'cmm' else r['id']
            d[k] = (g, p)
        arms[label] = d
        common = [k for k in d if k in base_bin]
        print(f'{label} (n={len(common)}):',
              metrics([d[k] for k in common]))

    # paired comparisons vs base
    print('\n== paired vs base ==')
    for label, d in arms.items():
        common = [k for k in d if k in base_bin]
        a = [d[k][1] for k in common]
        b = [base_bin[k][1] for k in common]
        yt = [d[k][0] for k in common]
        yn = int(sum(1 for x, y in zip(a, b) if y == 1 and x == 0))   # Y->N
        ny = int(sum(1 for x, y in zip(a, b) if y == 0 and x == 1))   # N->Y
        fix = int(sum(1 for x, y, g in zip(a, b, yt) if x != y and x == g))
        brk = int(sum(1 for x, y, g in zip(a, b, yt) if x != y and x != g))
        b01, b10, p = mcnemar(a, b, yt)
        print(f'{label}: N->Y {ny}, Y->N {yn}, fix {fix}, break {brk}, '
              f'McNemar {b01}:{b10} p={p:.2e}')

    # pairwise between arms
    labels = list(arms)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            da, db = arms[labels[i]], arms[labels[j]]
            common = [k for k in da if k in db]
            a = [da[k][1] for k in common]
            b = [db[k][1] for k in common]
            yt = [da[k][0] for k in common]
            b01, b10, p = mcnemar(a, b, yt)
            print(f'{labels[i]} vs {labels[j]}: McNemar {b01}:{b10} p={p:.2e}')


def score_file_rows(rows):
    """Attach eval_metrics using the repo scorer (in memory)."""
    ev = UnifiedEvaluator()
    for item in rows:
        gt_val = str(item.get('gt', ''))
        gt_lower = gt_val.lower().strip()
        if gt_lower == 'yes' or gt_lower.startswith(('yes.', 'yes,', 'yes ')):
            g = 1
        elif gt_lower == 'no' or gt_lower.startswith(('no.', 'no,', 'no ')):
            g = 0
        else:
            g = ev._parse_yesno_answer(gt_val)
        pred = ev._parse_yesno_answer(str(item.get('pred', item.get('response', ''))))
        undecidable = (g == 2)
        item['eval_metrics'] = {
            'Ground_Truth': g, 'Prediction': pred,
            'Is_Correct': 0 if (undecidable or g != pred) else 1,
        }
    return rows


if __name__ == '__main__':
    main()
