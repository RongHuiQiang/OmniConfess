#!/usr/bin/env python3
"""
Build blind-spot deliverables: reliability-curve family figure, BSI summary
(md + json), from confidence_blindspot.analyze over all scored files.

Headline口径 (seed-faithful):
  - Judgment tasks: surface confidence = answer margin (|cf.margin_full| or
    |lp_yes-lp_no|). This is THE correct surface-confidence lens for a binary
    judgment (how strongly the model prefers its answer over the alternative).
  - Generation tasks: surface confidence = norm_logp (sequence mean logprob),
    correctness = Token-F1>=median (binarized; noted as an approximation).

The figure overlays one line per selected (dataset x backbone x method) cell,
x = within-bin mean confidence min-max normalized to [0,1] for overlay,
y = within-bin accuracy. Diagonal reference drawn. Legend tags each BSI.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import confidence_blindspot as cb

OUT_DIR = "/home/tsq/rhq/omniconfess/outputs/problem_structure_fig"
os.makedirs(OUT_DIR, exist_ok=True)

BLIND_THRESH = 0.15


def pick_headline(results):
    """One representative cell per (dataset, backbone), preferring seed-faithful
    margin口径 for judgment tasks and base_model for generation tasks."""
    # datasets whose seed-faithful口径 is the answer margin (judgment tasks)
    def rank(r):
        f = r["conf_field"]
        tier = 3
        if f.startswith("|cf.margin"):
            tier = 0
        elif f.startswith("|lp_yes"):
            tier = 1
        elif r["method"] == "base_model":
            tier = 2
        # within a tier, prefer the largest, "full"/main run
        main = 0 if ("promptdev" not in r["method"] and "randctl" not in r["method"]
                     and "_v1" not in r["method"] and "invalid" not in r["method"]) else 1
        return (tier, main, -r["n"])
    best = {}
    for r in results:
        key = (r["dataset"], r["backbone"])
        if key not in best or rank(r) < rank(best[key]):
            best[key] = r
    return best


def main():
    results = cb.main()

    # drop alias-file duplicates (same dataset/backbone/method/n/bsi/rho)
    seen = set()
    deduped = []
    for r in results:
        key = (r["dataset"], r["backbone"], r["method"], r["n"], r["bsi"], r["rho"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    results = deduped

    # sort by BSI ascending (blind -> calibrated)
    results.sort(key=lambda x: x["bsi"])

    # ---- summary json ----
    with open(os.path.join(OUT_DIR, "bsi_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- summary md ----
    n_blind = sum(1 for r in results if r["bsi"] < BLIND_THRESH)
    n_cal = len(results) - n_blind
    lines = []
    lines.append("# Confidence Blind-Spot Index (BSI) summary\n")
    lines.append(f"Cells computed: **{len(results)}**  |  "
                 f"blind (BSI<{BLIND_THRESH}): **{n_blind}**  |  "
                 f"calibrated (BSI>={BLIND_THRESH}): **{n_cal}**\n")
    lines.append("BSI = acc(top-conf bin) - acc(bottom-conf bin). "
                 "rho = Spearman(conf, correct). Sorted ascending (blindest first).\n")
    lines.append("| dataset | backbone | method | task | n | conf field | n_bins | BSI | rho | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        verdict = "BLIND" if r["bsi"] < BLIND_THRESH else "calibrated"
        lines.append(f"| {r['dataset']} | {r['backbone']} | {r['method']} | "
                     f"{r['task_type']} | {r['n']} | {r['conf_field']} | "
                     f"{r['n_bins']} | {r['bsi']:+.3f} | {r['rho']:+.3f} | {verdict} |")
    with open(os.path.join(OUT_DIR, "bsi_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- figure: headline cells (one per dataset x backbone) ----
    headline = pick_headline(results)
    hitems = sorted(headline.values(), key=lambda x: x["bsi"])

    plt.figure(figsize=(9, 7))
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="0.6", label="perfect calibration ref")

    cmap = plt.get_cmap("coolwarm")
    for r in hitems:
        curve = np.array(r["curve"], float)  # (conf, acc, n)
        x = curve[:, 0]
        # min-max normalize confidence to [0,1] for overlay
        if x.max() > x.min():
            xn = (x - x.min()) / (x.max() - x.min())
        else:
            xn = np.zeros_like(x)
        y = curve[:, 1]
        # color: blind (low BSI) -> blue-ish, calibrated -> red-ish
        c = cmap(min(1.0, max(0.0, (r["bsi"] + 0.05) / 0.45)))
        lw = 2.6 if r["bsi"] < BLIND_THRESH else 1.6
        label = (f"{r['dataset']}/{r['backbone']} "
                 f"BSI={r['bsi']:+.2f}"
                 + (" [BLIND]" if r["bsi"] < BLIND_THRESH else ""))
        ax.plot(xn, y, "-", color=c, lw=lw, marker="o", ms=4, label=label)

    ax.set_xlabel("within-bin surface confidence (min-max normalized per cell)")
    ax.set_ylabel("within-bin accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(-0.02, 1.02)
    ax.set_title("Confidence-correctness decoupling on evidence-dispersed / multimodal tasks\n"
                 "flat lines = surface confidence blind to correctness (BSI~0); "
                 "rising = calibrated", fontsize=10)
    ax.legend(fontsize=7, loc="lower right", ncol=1, framealpha=0.9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_confidence_blindspot.png"), dpi=140)
    print("wrote figure + tables to", OUT_DIR)
    print(f"cells={len(results)} blind={n_blind} calibrated={n_cal}")
    print("headline cells:")
    for r in hitems:
        print(f"  {r['dataset']:9s} {r['backbone']:8s} BSI={r['bsi']:+.3f} "
              f"rho={r['rho']:+.3f} [{r['conf_field']}] {r['method']}")


if __name__ == "__main__":
    main()
