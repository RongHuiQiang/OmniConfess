#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_dependency.py — 离线统计 haloquest 每个 token 的依赖类型分布

三类:
  under_reliance  E_R < 0: 相关通道证据反对这个 token，模型依赖相关通道不足
  over_reliance   g3 > 0:  无关通道在驱动这个 token，模型过度依赖无关通道
  both            两者同时 (最严重的幻觉信号)
  normal          两者均不成立 (依赖关系正常)

不需要 GPU，纯离线。

用法:
  python3 diagnose_dependency.py [--tau TAU] [--dataset haloquest]
"""

import argparse
import json
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import freeform_repair as fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="haloquest")
    ap.add_argument("--tau", type=float, default=0.0,
                    help="依赖不足/过度的判定阈值 (默认0: >0即为异常)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--examples", type=int, default=5,
                    help="打印多少个含幻觉 token 的样本示例")
    args = ap.parse_args()

    print(f"[load] dataset={args.dataset} tau={args.tau}", flush=True)
    loader = fr.LOADERS[args.dataset]
    samples = loader()
    if args.limit > 0:
        samples = samples[:args.limit]
    print(f"[load] {len(samples)} samples", flush=True)

    cfg = fr.RepairConfig(tau=args.tau, select_mode="rho")

    # 全局计数
    total_tok = 0
    cnt = {"normal": 0, "under_reliance": 0, "over_reliance": 0, "both": 0}
    # 每样本有幻觉 token 的数量
    sample_rows = []

    for s in samples:
        scored = fr.score_sample(s, cfg)
        labels = scored["dep_labels"]   # per-token np array of strings
        n = len(labels)
        total_tok += n
        for lab in labels:
            cnt[lab] += 1

        n_under = int((labels == "under_reliance").sum())
        n_over  = int((labels == "over_reliance").sum())
        n_both  = int((labels == "both").sum())
        n_hal   = n_under + n_over + n_both

        sample_rows.append({
            "id": s["id"],
            "n_tokens": n,
            "n_under": n_under,
            "n_over": n_over,
            "n_both": n_both,
            "n_hal": n_hal,
            "frac_hal": n_hal / n if n else 0.0,
            "tokens": s["tokens"],
            "labels": labels,
            "g1": scored["gamma"]["g1"],
            "g3": scored["gamma"]["g3"],
        })

    # ── 全局统计 ──────────────────────────────────────────────────────────────
    print("\n===== GLOBAL TOKEN DEPENDENCY DISTRIBUTION =====")
    print(f"Total tokens: {total_tok}")
    for lab in ["normal", "under_reliance", "over_reliance", "both"]:
        pct = 100.0 * cnt[lab] / total_tok if total_tok else 0.0
        print(f"  {lab:<16}: {cnt[lab]:6d}  ({pct:.2f}%)")

    n_hal_global = cnt["under_reliance"] + cnt["over_reliance"] + cnt["both"]
    pct_hal = 100.0 * n_hal_global / total_tok if total_tok else 0.0
    print(f"  {'hallucination':16}: {n_hal_global:6d}  ({pct_hal:.2f}%)  [under+over+both]")

    if n_hal_global > 0:
        print("\n  Breakdown of hallucination tokens:")
        for lab in ["under_reliance", "over_reliance", "both"]:
            pct = 100.0 * cnt[lab] / n_hal_global
            print(f"    {lab:<16}: {pct:.1f}%")

    # ── 样本级分布 ────────────────────────────────────────────────────────────
    n_clean  = sum(1 for r in sample_rows if r["n_hal"] == 0)
    n_has    = sum(1 for r in sample_rows if r["n_hal"] > 0)
    frac_avg = np.mean([r["frac_hal"] for r in sample_rows])
    print(f"\n  Samples with ≥1 hallucination token: {n_has}/{len(sample_rows)} ({100.*n_has/len(sample_rows):.1f}%)")
    print(f"  Clean samples (0 hal tokens):        {n_clean}/{len(sample_rows)}")
    print(f"  Mean hallucination token frac/sample: {100.*frac_avg:.2f}%")

    # ── 示例 ──────────────────────────────────────────────────────────────────
    if args.examples > 0:
        examples = sorted(
            [r for r in sample_rows if r["n_hal"] > 0],
            key=lambda r: -r["frac_hal"]
        )[:args.examples]

        print(f"\n===== TOP {len(examples)} SAMPLES BY HAL FRACTION =====")
        for ex in examples:
            print(f"\n  id={ex['id']}  n_tok={ex['n_tokens']}  "
                  f"under={ex['n_under']} over={ex['n_over']} both={ex['n_both']}  "
                  f"frac={100.*ex['frac_hal']:.1f}%")
            # 打印每个 token 及其标签
            lines = []
            for i, (tok, lab) in enumerate(zip(ex["tokens"], ex["labels"])):
                if lab != "normal":
                    g1v = float(ex["g1"][i])
                    g3v = float(ex["g3"][i])
                    lines.append(f"    [{i:3d}] {tok!r:<20} {lab:<16}  g1={g1v:+.3f}  g3={g3v:+.3f}")
            for l in lines[:20]:  # 最多显示 20 个幻觉 token
                print(l)
            if len(lines) > 20:
                print(f"    ... ({len(lines)-20} more hallucination tokens)")

    # ── under vs over 的 g1/g3 幅度分布 ──────────────────────────────────────
    under_g1_vals = []
    over_g3_vals  = []
    both_g1_vals  = []
    both_g3_vals  = []

    for r in sample_rows:
        for i, lab in enumerate(r["labels"]):
            if lab == "under_reliance":
                under_g1_vals.append(float(r["g1"][i]))
            elif lab == "over_reliance":
                over_g3_vals.append(float(r["g3"][i]))
            elif lab == "both":
                both_g1_vals.append(float(r["g1"][i]))
                both_g3_vals.append(float(r["g3"][i]))

    print("\n===== SIGNAL MAGNITUDE (g1 / g3) =====")
    if under_g1_vals:
        arr = np.array(under_g1_vals)
        print(f"  under_reliance  g1  mean={arr.mean():+.4f}  "
              f"p25={np.percentile(arr,25):+.4f}  p50={np.percentile(arr,50):+.4f}  "
              f"p75={np.percentile(arr,75):+.4f}  min={arr.min():+.4f}")
    if over_g3_vals:
        arr = np.array(over_g3_vals)
        print(f"  over_reliance   g3  mean={arr.mean():+.4f}  "
              f"p25={np.percentile(arr,25):+.4f}  p50={np.percentile(arr,50):+.4f}  "
              f"p75={np.percentile(arr,75):+.4f}  max={arr.max():+.4f}")
    if both_g1_vals:
        arr = np.array(both_g1_vals)
        print(f"  both            g1  mean={arr.mean():+.4f}  "
              f"p50={np.percentile(arr,50):+.4f}  min={arr.min():+.4f}")
        arr = np.array(both_g3_vals)
        print(f"  both            g3  mean={arr.mean():+.4f}  "
              f"p50={np.percentile(arr,50):+.4f}  max={arr.max():+.4f}")


if __name__ == "__main__":
    main()
