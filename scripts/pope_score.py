#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pope_score.py -- Score POPE arms (base / cf / vcd / cad) in judgement-family
format: per setting (random / popular / adversarial) + overall
accuracy, precision, recall, F1 (positive class = yes), yes-rate. POPE standard.

Extra duties:
  --rule-lambda  compute the Z12 rule lambda_hat = 4.41 - 0.0477 * base_yes_rate
                 from a base jsonl (whole-run yes rate) and print it
  --oracle       sweep lambda over a grid on a cf jsonl's recorded
                 (margin_full, delta_cf) and report the best accuracy
                 (per setting and overall) with its lambda*

Usage:
  python scripts/pope_score.py --dir outputs/pope --pattern 'pope_base_*.jsonl'
  python scripts/pope_score.py --rule-lambda outputs/pope/pope_base_random.jsonl ...
  python scripts/pope_score.py --oracle outputs/pope/pope_cf_lam1.7_random.jsonl
"""
import argparse, glob, json, os, re, sys


def parse_yesno(text):
    t = str(text).lower().strip()
    m = re.search(r'\\boxed\{(.*?)\}', t)
    if m:
        t = m.group(1)
    # first yes/no occurrence
    m = re.search(r'\b(yes|no)\b', t)
    if m:
        return m.group(1)
    return 'yes' if t.startswith('y') else ('no' if t.startswith('n') else 'na')


def load(path):
    items = []
    with open(path) as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items


def scores(items, pred_key='pred'):
    tp = fp = fn = tn = na = 0
    for it in items:
        p = parse_yesno(it.get(pred_key) if isinstance(it.get(pred_key), str)
                        else it.get('response', it.get('raw_output', '')))
        g = str(it.get('gt', '')).lower().strip()
        if p == 'na':
            na += 1
            p = 'no'  # POPE convention: unparseable counted as no
        if p == 'yes' and g == 'yes':
            tp += 1
        elif p == 'yes':
            fp += 1
        elif g == 'yes':
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    yr = (tp + fp) / n if n else 0.0
    return dict(n=n, acc=round(acc * 100, 2), precision=round(prec * 100, 2),
                recall=round(rec * 100, 2), f1=round(f1 * 100, 2),
                yes_rate=round(yr * 100, 2), unpar=na)


def by_setting(items):
    groups = {}
    for it in items:
        s = it.get('pope_setting') or it.get('group_key') or '?'
        groups.setdefault(s, []).append(it)
    return groups


def report(path):
    items = load(path)
    groups = by_setting(items)
    print(f'\n== {os.path.basename(path)} ==')
    print(f'{"setting":<14}{"n":>5}{"acc":>8}{"prec":>8}{"rec":>8}{"F1":>8}{"yes%":>8}')
    for s in ['random', 'popular', 'adversarial']:
        if s in groups:
            r = scores(groups[s])
            print(f'{s:<14}{r["n"]:>5}{r["acc"]:>8}{r["precision"]:>8}{r["recall"]:>8}{r["f1"]:>8}{r["yes_rate"]:>8}')
    r = scores(items)
    print(f'{"overall":<14}{r["n"]:>5}{r["acc"]:>8}{r["precision"]:>8}{r["recall"]:>8}{r["f1"]:>8}{r["yes_rate"]:>8}')
    return r


def rule_lambda(paths):
    tot = yes = 0
    for p in paths:
        for it in load(p):
            tot += 1
            pr = parse_yesno(it.get('pred') if isinstance(it.get('pred'), str)
                             else it.get('response', ''))
            yes += (pr == 'yes')
    rate = yes / tot if tot else 0
    lam = 4.41 - 0.0477 * rate
    print(f'[rule] base yes rate = {yes}/{tot} = {rate:.4f} -> lambda_hat = {lam:.4f}')
    return lam


def oracle(paths):
    """Sweep lambda on recorded (margin_full, delta_cf); decision Yes iff
    margin_full + lam*delta > 0. Uses each item's gt."""
    grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
            4.41, 5.0, 6.0, 8.0, 10.0, -1.0, -0.5]
    rows = []
    for p in paths:
        for it in load(p):
            cf = it.get('cf') or {}
            if cf.get('margin_full') is None or cf.get('delta_cf') is None:
                continue
            rows.append((cf['margin_full'], cf['delta_cf'],
                         str(it.get('gt', '')).lower().strip()))
    if not rows:
        print('[oracle] no cf margin records found')
        return
    print(f'[oracle] {len(rows)} usable cf records')
    best = None
    for lam in sorted(grid):
        acc = sum(1 for m, d, g in rows
                  if ('yes' if m + lam * d > 0 else 'no') == g) / len(rows)
        print(f'  lam={lam:>6}: acc={acc*100:.2f}')
        if best is None or acc > best[1]:
            best = (lam, acc)
    print(f'[oracle] best lam*={best[0]} acc={best[1]*100:.2f} '
          f'(grid search, {len(rows)} samples)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='outputs/pope')
    ap.add_argument('--pattern', default='*.jsonl')
    ap.add_argument('--rule-lambda', nargs='*', default=None)
    ap.add_argument('--oracle', nargs='*', default=None)
    args = ap.parse_args()

    if args.rule_lambda is not None:
        rule_lambda(args.rule_lambda or glob.glob(os.path.join(args.dir, 'pope_base_*.jsonl')))
    if args.oracle is not None:
        oracle(args.oracle or glob.glob(os.path.join(args.dir, 'pope_cf_*.jsonl')))

    for p in sorted(glob.glob(os.path.join(args.dir, args.pattern))):
        report(p)


if __name__ == '__main__':
    main()
