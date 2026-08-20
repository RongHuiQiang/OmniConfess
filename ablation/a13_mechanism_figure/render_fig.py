# -*- coding: utf-8 -*-
"""render_fig.py — paper §4.0 problem-mechanism figure (2x2, zero GPU).

Renders fig_mechanism.{pdf,png} from a13_fig2_panels.json, the panel CSVs and
the raw Qwen2.5 probe files, per PAPER_SEC4_FIG_DESIGN.md. Every annotated
number is read programmatically from those files; none is hand-typed.

Panels (organisation = our own four-question proof chain):
  (a) does the disease exist   — ECDF of |d| for wrong vs correct commitments
                                 (phd |delta_noctx|, cmm |d_cf_sum|; Qwen2.5
                                 full probes), AUROC annotated from json.
  (b) is it concentrated       — Lorenz curves of |d| mass (Gini + top-10%
                                 mass from json), with a flippable-band-vs-lambda
                                 inset (decision: inset kept, see RENDER_NOTES).
  (c) is it hidden             — IoU(top-p% |d|, top-p% |margin|) with bootstrap
                                 CI and random-overlap reference y = p.
  (d) life cycle               — slope chart across backbones; band% (left y)
                                 and median |d| (right y, log). Qwen2.5 point
                                 uses the dev sample (cross_backbone_dev node).

Style: colorblind-safe palette (dataset = color: phd blue #0072B2, cmm orange
#E69F00), correctness/metric = linestyle.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CF = os.path.join(ROOT, "outputs", "cf_probe")

C_PHD = "#0072B2"   # blue  (Okabe-Ito)
C_CMM = "#E69F00"   # orange (Okabe-Ito)
C_REF = "#999999"

J = json.load(open(os.path.join(HERE, "a13_fig2_panels.json")))
PA = J["panel_a_concentration"]
PC = J["panel_c_validity"]


def jl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ----------------------------------------------------------------- panel (a)
def panel_a(ax):
    phd = jl(os.path.join(CF, "phd_probe_full.jsonl"))
    cmm = jl(os.path.join(CF, "cmm_probe_full.jsonl"))
    auroc = PC["auroc_cited_from_a3"]
    series = [
        ("phd", C_PHD, "|$d_{cf}$| (no-context)", phd, "delta_noctx",
         auroc["phd_q2.5_full"]),
        ("cmm", C_CMM, "|$d_{cf}^{\\sum}$| (audio+video)", cmm, "d_cf_sum",
         auroc["cmm_q2.5_full"]),
    ]
    for name, col, lab, rows, key, au in series:
        d = np.abs(np.array([r[key] for r in rows], float))
        bc = np.array([r["base_correct"] for r in rows], int)
        for flag, style, slab in [(0, "--", "wrong"), (1, "-", "correct")]:
            x = np.sort(d[bc == flag])
            x = x[x > 0]  # log axis needs positives; zeros noted in caption
            y = np.arange(1, len(x) + 1) / len(d[bc == flag])
            ax.plot(x, y, style, color=col, lw=1.8,
                    label=f"{name} · {slab} (n={int((bc == flag).sum())})")
        ax.annotate(
            f"{name} AUROC {au['auc']:.3f} "
            f"[{au['ci'][0]:.3f}, {au['ci'][1]:.3f}]",
            xy=(0.97, 0.06 + 0.07 * (name == "cmm")),
            xycoords="axes fraction", ha="right", color=col, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("|d| (nats, log scale)")
    ax.set_ylabel("cumulative fraction of commitments")
    ax.set_title("(a) Wrong commitments carry stronger CF dependence",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.5)


# ----------------------------------------------------------------- panel (b)
def panel_b(ax):
    df = pd.read_csv(os.path.join(HERE, "a13_panel_a_lorenz.csv"))
    lg = PA["lorenz_gini"]
    specs = [
        ("phd_q2.5_full_|delta_noctx|", C_PHD, "phd",
         lg["phd_q2.5_full_|delta_noctx|"],
         lg["phd_delta_noctx_top_share"]),
        ("cmm_q2.5_full_|d_cf_sum|", C_CMM, "cmm",
         lg["cmm_q2.5_full_|d_cf_sum|"],
         lg["cmm_d_cf_sum_top_share"]),
    ]
    for cond, col, name, g, ts in specs:
        s = df[df.condition == cond].sort_values("pop_frac")
        ax.plot(s.pop_frac, s.mass_frac, "-", color=col, lw=2.0,
                label=(f"{name}: Gini {g['gini']:.3f} "
                       f"[{g['ci95'][0]:.3f}, {g['ci95'][1]:.3f}]"
                       f"; top-10% mass {ts['top10pct_mass']*100:.1f}%"))
    ax.plot([0, 1], [0, 1], "--", color=C_REF, lw=1.2, label="diffuse reference")
    ax.set_xlabel("cumulative share of commitment points (ranked by |d|)")
    ax.set_ylabel("cumulative share of |d| mass")
    ax.set_title("(b) Dependence mass concentrates on few commitments",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # inset: flippable-band share vs lambda (render-time decision: kept)
    fb = PA["flippable_band"]
    ins = ax.inset_axes([0.42, 0.06, 0.30, 0.34])
    lams = sorted(float(k) for k in fb["phd"])
    for fam, col in [("phd", C_PHD), ("cmm", C_CMM)]:
        v = [fb[fam][str(l)] * 100 for l in lams]
        ci = np.array([[fb[fam + "_ci"][str(l)][0] * 100,
                        fb[fam + "_ci"][str(l)][1] * 100] for l in lams]).T
        ins.errorbar(lams, v, yerr=[v - ci[0], ci[1] - v], marker="o",
                     ms=3, color=col, lw=1.4, capsize=2,
                     label=f"{fam} $\\lambda$=2: {fb[fam]['2.0']*100:.1f}%")
    ins.set_xscale("log", base=2)
    ins.set_xticks(lams); ins.set_xticklabels(["0.5", "1", "2", "4"])
    ins.set_xlabel("$\\lambda$", fontsize=7); ins.tick_params(labelsize=6.5)
    ins.set_ylabel("flippable band (%)", fontsize=7)
    ins.legend(fontsize=6, loc="lower right", framealpha=0.9)
    ins.set_title("|margin| < $\\lambda$|d|", fontsize=7)


# ----------------------------------------------------------------- panel (c)
def panel_c(ax):
    df = pd.read_csv(os.path.join(HERE, "a13_panel_c_iou.csv"))
    for name, col in [("phd", C_PHD), ("cmm", C_CMM)]:
        s = df[df.dataset == name].sort_values("pct")
        v20 = float(s[s.pct == 20].iou.iloc[0])
        ax.plot(s.pct, s.iou, "o-", color=col, lw=1.8, ms=4,
                label=f"{name}: IoU@20% = {v20:.3f}")
        ax.fill_between(s.pct, s.ci_lo, s.ci_hi, color=col, alpha=0.18, lw=0)
    p = np.sort(df.pct.unique())
    ax.plot(p, p / 100.0, "--", color=C_REF, lw=1.2,
            label="random overlap (= p)")
    ax.set_xscale("log")
    ax.set_xticks(p); ax.set_xticklabels([f"{x}%" for x in p])
    ax.set_xlabel("top-p% threshold")
    ax.set_ylabel("IoU(top-p% |d|, top-p% |margin|)")
    ax.set_title("(c) Dependence is orthogonal to model uncertainty",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5)


# ----------------------------------------------------------------- panel (d)
def panel_d(ax):
    xb = PA["cross_backbone"]
    dev = PA["cross_backbone_dev"]
    names = ["Qwen2.5\n(dev)", "Qwen3\n(dev)", "Nemotron\n(dev)"]
    specs = []
    for fam, col in [("phd", C_PHD), ("cmm", C_CMM)]:
        band = [dev[fam]["band_lam2"] * 100, xb[fam]["qwen3_dev"]["band_lam2"] * 100,
                xb[fam]["nemotron_dev"]["band_lam2"] * 100]
        med = [dev[fam]["median_abs_d"], xb[fam]["qwen3_dev"]["median_abs_d"],
               xb[fam]["nemotron_dev"]["median_abs_d"]]
        n = [dev[fam]["n"], xb[fam]["qwen3_dev"]["n"],
             xb[fam]["nemotron_dev"]["n"]]
        specs.append((fam, col, band, med, n))

    ax2 = ax.twinx()
    for fam, col, band, med, n in specs:
        ax.plot(range(3), band, "o-", color=col, lw=2.0, ms=5,
                label=f"{fam}: band% (n={n[0]}/{n[1]}/{n[2]})")
        ax2.plot(range(3), med, ":", color=col, lw=1.8, marker="s", ms=4,
                 alpha=0.75)
        for i, (b, m) in enumerate(zip(band, med)):
            ax.annotate(f"{b:.1f}", (i, b), textcoords="offset points",
                        xytext=(0, 7), ha="center", color=col, fontsize=7.5)
            ax2.annotate(f"{m:.3g}", (i, m), textcoords="offset points",
                         xytext=(0, -11), ha="center", color=col, fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels(names)
    ax.set_ylabel("flippable band @ $\\lambda$=2 (%)")
    ax2.set_ylabel("median |d| (nats, log)", rotation=270,
                   labelpad=14)
    ax2.set_yscale("log")
    ax.set_title("(d) Pathology self-heals across backbone generations",
                 fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    ax.plot([], [], ":", color="k", alpha=0.75, label="median |d| (right, log)")
    ax.legend(fontsize=7.5, loc="center right", framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_ylim(-8, None)


# --------------------------------------------------------------------- main
def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(HERE, f"fig_mechanism.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
