#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_haloquest_armd_full365.py -- offline Token-F1 evaluation for the
ArmD (κ^image confession) pipeline on the full HaloQuest 365-sample set.

Scorer is identical to tokenf1_fullset.py / single_eval.py:
  - lowercase + strip punctuation
  - remove STOPWORDS (38 words)
  - Counter intersection F1

Usage:
    python scripts/eval_haloquest_armd_full365.py \
        outputs/lv2_full365/confession_full365_merged.jsonl

Outputs per-sample CSV to stdout if --csv flag is set, else prints a summary.
"""

import argparse
import json
import math
import random
import string
import sys
from collections import Counter
from pathlib import Path

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "are", "was", "were",
    "of", "while", "during", "to", "in", "on", "at", "by", "with", "it", "there"
}

TARGET_BASE = 39.37
TARGET_ARMD = 43.52


def _tokens(s):
    s = str(s).lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in s.split() if w not in STOPWORDS and w.strip()]


def token_f1(pred, gt):
    p_t, g_t = _tokens(pred), _tokens(gt)
    if not p_t or not g_t:
        return 0.0
    common = Counter(p_t) & Counter(g_t)
    hit = sum(common.values())
    p = hit / len(p_t)
    r = hit / len(g_t)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def bootstrap_ci(scores, n_boot=2000, alpha=0.05, seed=42):
    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(n_boot):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return lo, hi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default="outputs/lv2_full365/confession_full365_merged.jsonl",
        help="Path to confession_full365_merged.jsonl",
    )
    parser.add_argument("--csv", action="store_true", help="Dump per-sample CSV")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        sys.exit(f"File not found: {path}")

    rows = [json.loads(l) for l in path.open() if l.strip()]
    n_total = len(rows)

    base_f1s, rep_f1s, edited_flags, ids = [], [], [], []
    for r in rows:
        gt = r.get("gt", "")
        bp = r.get("base_pred", "")
        rp = r.get("repaired_pred", bp)  # fall back to base if missing
        base_f1s.append(token_f1(bp, gt))
        rep_f1s.append(token_f1(rp, gt))
        edited_flags.append(bool(r.get("edited", False)))
        ids.append(r.get("id", ""))

    n_edited = sum(edited_flags)
    mean_base = 100 * sum(base_f1s) / n_total
    mean_rep = 100 * sum(rep_f1s) / n_total
    delta = mean_rep - mean_base

    delta_per = [100 * (r - b) for r, b in zip(rep_f1s, base_f1s)]
    lo_d, hi_d = bootstrap_ci(delta_per)

    # edited-only subset
    edited_base = [b for b, e in zip(base_f1s, edited_flags) if e]
    edited_rep = [r for r, e in zip(rep_f1s, edited_flags) if e]
    if edited_base:
        mean_base_ed = 100 * sum(edited_base) / len(edited_base)
        mean_rep_ed = 100 * sum(edited_rep) / len(edited_rep)
        delta_ed = mean_rep_ed - mean_base_ed
        delta_per_ed = [100 * (r - b) for r, b in zip(edited_rep, edited_base)]
        lo_ed, hi_ed = bootstrap_ci(delta_per_ed)
    else:
        mean_base_ed = mean_rep_ed = delta_ed = lo_ed = hi_ed = float("nan")

    if args.csv:
        print("id,gt,base_f1,rep_f1,delta,edited")
        for i, (r, bf, rf, ef) in enumerate(zip(rows, base_f1s, rep_f1s, edited_flags)):
            print(f"{ids[i]},{100*bf:.4f},{100*rf:.4f},{100*(rf-bf):.4f},{int(ef)}")
        return

    print("=" * 60)
    print("ArmD full-365 Token-F1 evaluation")
    print(f"  Input : {path}")
    print(f"  N     : {n_total}")
    print("=" * 60)
    print(f"  base_pred  Token-F1 : {mean_base:.2f}  (target baseline {TARGET_BASE})")
    print(f"  ArmD_pred  Token-F1 : {mean_rep:.2f}  (target >= {TARGET_ARMD})")
    print(f"  Delta                : {delta:+.2f}  95% CI [{lo_d:+.2f}, {hi_d:+.2f}]")
    print()
    print(f"  Edited samples       : {n_edited} / {n_total} ({100*n_edited/n_total:.1f}%)")
    if not math.isnan(delta_ed):
        print(f"  Edited-only delta    : {delta_ed:+.2f}  95% CI [{lo_ed:+.2f}, {hi_ed:+.2f}]")
    print()
    if mean_rep >= TARGET_ARMD:
        print(f"  PASS: ArmD Token-F1 {mean_rep:.2f} >= target {TARGET_ARMD}")
    else:
        print(f"  FAIL: ArmD Token-F1 {mean_rep:.2f} < target {TARGET_ARMD}")
    print("=" * 60)


if __name__ == "__main__":
    main()
