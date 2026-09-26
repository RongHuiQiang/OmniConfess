#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cf_override_summary.py -- CF_OVERRIDE (2026-08-31) per-arm summary.

Reads one *_single_scored.jsonl from a CF_OVERRIDE cell and reports:
  - n, yes-rate, F1 / PhD-Index / Youden / FPR / recalls (recomputed from
    eval_metrics, official scorer convention: maybe-pred = miss)
  - CF internals: trigger rate, direction split (Yes->No / No->Yes),
    delta_cf distribution, margin distribution
  - in-run paired base comparison: the runner records the base greedy
    decision (pred_base / greedy_token; qwen3 arm: greedy_token) on every
    sample, so fix/break/McNemar vs the arm's own base trajectory is exact
    (the old base files lived on /ultrafast_share, wiped 2026-08-29).

Usage: python cf_override_summary.py <scored.jsonl> [--qwen3]
"""
import json, sys, argparse
import math


def f1_pr(yt, yp):
    tp = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(yt, yp) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(yt, yp) if a == 0 and b == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    r_no = tn / (tn + fp) if tn + fp else 0.0
    youden = rec + r_no - 1
    fpr = fp / (fp + tn) if fp + tn else 0.0
    idx = 2 * rec * r_no / (rec + r_no) if rec + r_no else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall_yes=rec,
                recall_no=r_no, F1=100 * f1, PhD_Index=100 * idx,
                Youden=youden, FPR=fpr)


def mcnemar(b, c):
    """Exact binomial McNemar p (two-sided) on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * p)


def is_yes(t):
    """Decision-token ids: scorer 1; Qwen2.5/Qwen3 9454; Nemotron 16860."""
    return t in (1, 9454, 16860)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scored')
    ap.add_argument('--qwen3', action='store_true',
                    help='arm A (cf_confess_qwen3.py record layout)')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.scored) if l.strip()]
    yt, yp = [], []
    for r in rows:
        m = r['eval_metrics']
        if m.get('Undecidable'):
            continue
        yt.append(m['Ground_Truth'])
        # maybe-pred = miss: flip away from gt (official scorer convention)
        p = m['Prediction']
        yp.append((1 - m['Ground_Truth']) if p == 2 else p)

    m = f1_pr(yt, yp)
    print(f"== {args.scored.split('/')[-1]}  n={len(yt)} "
          f"yes-rate={100*sum(yp)/len(yp):.1f}%")
    print(f"   F1={m['F1']:.2f}  PhD-Index={m['PhD_Index']:.2f}  "
          f"Youden={m['Youden']:.3f}  FPR={m['FPR']:.3f}  "
          f"r_yes={m['recall_yes']:.3f} r_no={m['recall_no']:.3f}  "
          f"(tp{m['tp']}/fp{m['fp']}/fn{m['fn']}/tn{m['tn']})")

    cf = [r.get('cf') or {} for r in rows]
    have = [c for c in cf if c.get('dec_index') is not None]
    interv = [c for c in have if c.get('intervened')]
    yn = sum(1 for c in interv
             if is_yes(c.get('greedy_token')) and not is_yes(c.get('forced_token')))
    ny = sum(1 for c in interv
             if not is_yes(c.get('greedy_token')) and is_yes(c.get('forced_token')))
    deltas = [c['delta_cf'] for c in have if isinstance(c.get('delta_cf'), (int, float))]
    deltas = sorted(deltas)
    margins = sorted(abs(c['margin_full']) for c in have
                     if isinstance(c.get('margin_full'), (int, float)))
    nodec = sum(1 for c in cf if c.get('dec_index') is None)
    errs = sum(1 for c in cf if c.get('error'))

    def med(v):
        return v[len(v) // 2] if v else float('nan')

    print(f"   decision-token located: {len(have)}/{len(rows)} (no-dec {nodec}, "
          f"errors {errs})")
    print(f"   trigger rate: {len(interv)}/{len(have)} = "
          f"{100*len(interv)/max(len(have),1):.1f}%   "
          f"direction Yes->No {yn} / No->Yes {ny}")
    if deltas:
        print(f"   delta_cf: median={med(deltas):+.3f} mean={sum(deltas)/len(deltas):+.3f} "
              f"share>0={100*sum(1 for d in deltas if d > 0)/len(deltas):.0f}% "
              f"max|d|={max(abs(d) for d in deltas):.2f}")
    if margins:
        print(f"   |margin_full| median={med(margins):.2f} "
              f"(error rows sit near the boundary if small)")

    # in-run paired base comparison (base greedy decision recorded per sample)
    yt_b, yp_b, yp_f, ok = [], [], [], 0
    for r, c in zip(rows, cf):
        mrec = r['eval_metrics']
        if mrec.get('Undecidable'):
            continue
        gt = mrec['Ground_Truth']
        p_fin = mrec['Prediction']
        p_fin = (1 - gt) if p_fin == 2 else p_fin
        gb = c.get('pred_base')
        if gb is None and c.get('greedy_token') is not None:
            gb = 'yes' if is_yes(c['greedy_token']) else 'no'
        if gb not in ('yes', 'no'):
            continue
        pb = 1 if gb == 'yes' else 0
        yt_b.append(gt); yp_b.append(pb); yp_f.append(p_fin); ok += 1
    if ok:
        mb = f1_pr(yt_b, yp_b)
        fixed = sum(1 for g, b, f in zip(yt_b, yp_b, yp_f) if b != g and f == g)
        broken = sum(1 for g, b, f in zip(yt_b, yp_b, yp_f) if b == g and f != g)
        print(f"   in-run base (greedy decision, n={ok}): "
              f"F1={mb['F1']:.2f} Youden={mb['Youden']:.3f} "
              f"yes-rate={100*sum(yp_b)/len(yp_b):.1f}%")
        print(f"   paired vs in-run base: fix {fixed} / break {broken} "
              f"(McNemar exact p={mcnemar(broken, fixed):.2g})")


if __name__ == '__main__':
    main()
