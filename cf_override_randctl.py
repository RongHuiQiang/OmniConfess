#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cf_override_randctl.py -- offline random-sign control (randctl) for the
CF_OVERRIDE arms (2026-08-31, coordinator request #1).

WHY OFFLINE IS EXACT (no GPU): the operator's scored decision is a hard binary
at the decision token -- margin' = margin_full + lam * (s * delta_cf) decides
Yes/No, and the final parsed prediction IS that token (verified: pred ==
forced_token on every decision-located row of all five runs on disk, including
the actually-rerun lam4.0 randctl GPU arm: 0 mismatches in ~4.5k rows). The
regeneration after forcing only rewrites the free-text tail AFTER the decision
token, which the scorer never reads differently (first \\boxed{} match is the
decision). So a randctl arm is pure arithmetic on the recorded cf fields.

Protocol (byte-copies cf_confess_phd/cmm/qwen3.py + cf_confess_nemotron_hf.py):
    control_rng = random.Random(seed + 7)     # seed=42 default -> 49
    per SAMPLE WITH A LOCATED DECISION TOKEN, in processing order:
        s = control_rng.choice([-1.0, 1.0])   # one draw per dec row only
    margin' = margin_full + lam * s * delta_cf -> forced Yes/No
RNG stream structure matters when the original run was sharded (each shard
process restarts the stream): --shards 3 --order concat replicates the
lam4.0-era 3x302 round-robin layout (contiguous blocks in the merged file).

Validations printed before any permutation number is trusted:
  A. main-arm reconstruct (s=+1) must equal the official scored F1 exactly
  B. (optional --val-randctl-file) protocol randctl(49) reconstruct must equal
     the official F1 of an actually-rerun randctl GPU arm (lam4.0)

Permutation distribution: P independent seeds, single stream, F1 per seed;
reports mean/sd/p5/p95, P(F1_perm >= F1_main) (sign-null exceedance), mean net
gain vs the in-run base, and the protocol-exact randctl(49) arm's F1.

Usage:
  python cf_override_randctl.py --file <scored.jsonl> --lam 2.0 \
      [--shards N --order concat|data] [--perms 2000] \
      [--val-randctl-file <scored_randctl.jsonl>]
"""
import json, random, argparse, math


def f1_of(gt, pred):
    tp = fp = fn = 0
    for g, p in zip(gt, pred):
        if p == 1 and g == 1: tp += 1
        elif p == 1 and g == 0: fp += 1
        elif p == 0 and g == 1: fn += 1
    return 100.0 * 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def load_rows(path):
    rows = []
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        m = r['eval_metrics']
        if m.get('Undecidable'):
            continue
        rows.append(r)
    return rows


def prep(rows):
    """Split into decision-located rows (arithmetic) and pass-through rows."""
    dec, passthrough = [], []
    for r in rows:
        cf = r.get('cf') or {}
        gt = r['eval_metrics']['Ground_Truth']
        p_final = r['eval_metrics']['Prediction']
        p_final = (1 - gt) if p_final == 2 else p_final   # maybe=miss
        if cf.get('dec_index') is None or cf.get('margin_full') is None:
            passthrough.append((gt, p_final))
            continue
        dec.append(dict(gt=gt, margin=cf['margin_full'], delta=cf['delta_cf'],
                        greedy=1 if cf.get('greedy_token') in (1, 9454, 16860)
                        else 0,
                        forced=1 if cf.get('forced_token') in (1, 9454, 16860)
                        else 0))
    return dec, passthrough


def assign_signs(dec, shards, order, seed):
    """One Random(seed) stream PER SHARD PROCESS, drawn per dec row in that
    process's processing order (file order within its block)."""
    n = len(dec)
    if shards <= 1:
        rng = random.Random(seed)
        return [rng.choice([-1.0, 1.0]) for _ in range(n)]
    if order == 'concat':          # merged file = shard blocks back to back
        blocks = [range(k * (n // shards) + min(k, n % shards),
                        (k + 1) * (n // shards) + min(k + 1, n % shards))
                  for k in range(shards)]
    else:                          # data order: round-robin interleaved
        blocks = [range(k, n, shards) for k in range(shards)]
    signs = [1.0] * n
    for blk in blocks:
        rng = random.Random(seed)
        for i in blk:
            signs[i] = rng.choice([-1.0, 1.0])
    return signs


def preds_for(dec, lam, signs):
    out = []
    for d, s in zip(dec, signs):
        if s == 1.0:
            out.append(d['forced'])            # == margin+lam*delta>0 branch
        else:
            out.append(1 if d['margin'] - lam * d['delta'] > 0 else 0)
    return out


def preds_main(dec):
    return [d['forced'] for d in dec]


def preds_base(dec):
    return [d['greedy'] for d in dec]


def arm_stats(dec, passthrough, preds):
    gt = [d['gt'] for d in dec] + [g for g, _ in passthrough]
    p = preds + [pf for _, pf in passthrough]
    return f1_of(gt, p), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--lam', type=float, required=True)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--order', choices=['concat', 'data'], default='concat')
    ap.add_argument('--seed', type=int, default=42,
                    help='runner --seed; rng = seed+7 (protocol)')
    ap.add_argument('--perms', type=int, default=2000)
    ap.add_argument('--perm-seed-base', type=int, default=1000)
    ap.add_argument('--val-randctl-file', default=None)
    args = ap.parse_args()

    rows = load_rows(args.file)
    dec, passthrough = prep(rows)
    print(f'== {args.file.split("/")[-1]}  lam={args.lam}  '
          f'n={len(rows)} (dec {len(dec)} + passthrough {len(passthrough)})')

    f1_main, _ = arm_stats(dec, passthrough, preds_main(dec))
    f1_base, _ = arm_stats(dec, passthrough, preds_base(dec))
    official = f1_of([r['eval_metrics']['Ground_Truth'] for r in rows],
                     [(1 - r['eval_metrics']['Ground_Truth'])
                      if r['eval_metrics']['Prediction'] == 2
                      else r['eval_metrics']['Prediction'] for r in rows])
    print(f'   official F1={official:.2f} | reconstruct main F1={f1_main:.2f} '
          f'(validation A {"OK" if abs(f1_main - official) < 0.005 else "FAIL"})'
          f' | in-run base F1={f1_base:.2f}')

    # protocol-exact randctl arm (what a GPU rerun with --control random_sign
    # would score; seed 49 = 42+7)
    signs = assign_signs(dec, args.shards, args.order, args.seed + 7)
    f1_rc, p_rc = arm_stats(dec, passthrough, preds_for(dec, args.lam, signs))
    n_flip = sum(1 for d, p in zip(dec, p_rc) if d['greedy'] != p)
    yes_rate = 100 * sum(p_rc) / len(p_rc)
    print(f'   randctl(49) F1={f1_rc:.2f}  (vs main {f1_main - f1_rc:+.2f}, '
          f'vs base {f1_rc - f1_base:+.2f})  triggered-vs-base={n_flip} '
          f'yes-rate={yes_rate:.1f}%')

    if args.val_randctl_file:
        rr = load_rows(args.val_randctl_file)
        ddec, dpass = prep(rr)
        s2 = assign_signs(ddec, args.shards, args.order, args.seed + 7)
        f1_rc2, _ = arm_stats(ddec, dpass, preds_for(ddec, args.lam, s2))
        off2 = f1_of([r['eval_metrics']['Ground_Truth'] for r in rr],
                     [(1 - r['eval_metrics']['Ground_Truth'])
                      if r['eval_metrics']['Prediction'] == 2
                      else r['eval_metrics']['Prediction'] for r in rr])
        print(f'   [validation B vs real GPU randctl arm] '
              f'official={off2:.2f} reconstruct={f1_rc2:.2f} '
              f'({"OK" if abs(f1_rc2 - off2) < 0.005 else "FAIL"})')

    # permutation distribution (single stream, like a single-process rerun)
    gt_all = [d['gt'] for d in dec] + [g for g, _ in passthrough]
    pass_p = [pf for _, pf in passthrough]
    f1s = []
    gt_dec = [d['gt'] for d in dec]
    for j in range(args.perms):
        rng = random.Random(args.perm_seed_base + j)
        signs = [rng.choice([-1.0, 1.0]) for _ in range(len(dec))]
        p = preds_for(dec, args.lam, signs)
        tp = fp = fn = 0
        for g, q in zip(gt_dec, p):
            if q == 1 and g == 1: tp += 1
            elif q == 1 and g == 0: fp += 1
            elif q == 0 and g == 1: fn += 1
        for g, q in zip(passthrough, pass_p):
            if q == 1 and g == 1: tp += 1
            elif q == 1 and g == 0: fp += 1
            elif q == 0 and g == 1: fn += 1
        f1s.append(100.0 * 2 * tp / (2 * tp + fp + fn))
    f1s.sort()
    n = len(f1s)
    mean = sum(f1s) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in f1s) / (n - 1))
    p5, p95 = f1s[int(0.05 * n)], f1s[int(0.95 * n)]
    exceed = sum(1 for x in f1s if x >= f1_main) / n
    print(f'   perm dist (P={n}): mean={mean:.2f} sd={sd:.2f} '
          f'p5={p5:.2f} p95={p95:.2f} | mean net vs base={mean - f1_base:+.2f}'
          f' | P(F1_perm >= F1_main)={exceed:.3f}')
    print(f'   verdict: main-vs-base {f1_main - f1_base:+.2f} | '
          f'main-vs-randctl(49) {f1_main - f1_rc:+.2f} | '
          f'main-vs-perm-mean {f1_main - mean:+.2f}')


if __name__ == '__main__':
    main()
