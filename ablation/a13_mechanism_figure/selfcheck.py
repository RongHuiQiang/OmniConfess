# -*- coding: utf-8 -*-
"""selfcheck.py — dump every number annotated in fig_mechanism, recomputed from
the json/csv sources the renderer reads (no hand-typed values). Output is
pasted into RENDER_NOTES.md as the audit table.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CF = os.path.abspath(os.path.join(HERE, "..", "..", "outputs", "cf_probe"))
J = json.load(open(os.path.join(HERE, "a13_fig2_panels.json")))
PA, PC = J["panel_a_concentration"], J["panel_c_validity"]


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


rows = []

# (a)
au = PC["auroc_cited_from_a3"]
for fam, f, key in [("phd", "phd_probe_full.jsonl", "delta_noctx"),
                    ("cmm", "cmm_probe_full.jsonl", "d_cf_sum")]:
    r = jl(os.path.join(CF, f))
    bc = np.array([x["base_correct"] for x in r])
    a = au[f"{'phd' if fam=='phd' else 'cmm'}_q2.5_full"]
    rows.append((f"(a) {fam} AUROC", f"{a['auc']:.3f} [{a['ci'][0]:.3f},"
                 f" {a['ci'][1]:.3f}]", "auroc_cited_from_a3 (a3)"))
    rows.append((f"(a) {fam} n wrong/correct",
                 f"{int((bc==0).sum())}/{int((bc==1).sum())}",
                 "probe_full.jsonl base_correct counts"))

# (b)
lg = PA["lorenz_gini"]
for fam, gk, tk in [("phd", "phd_q2.5_full_|delta_noctx|",
                     "phd_delta_noctx_top_share"),
                    ("cmm", "cmm_q2.5_full_|d_cf_sum|", "cmm_d_cf_sum_top_share")]:
    g, t = lg[gk], lg[tk]
    rows.append((f"(b) {fam} Gini",
                 f"{g['gini']:.3f} [{g['ci95'][0]:.3f}, {g['ci95'][1]:.3f}]",
                 "lorenz_gini"))
    rows.append((f"(b) {fam} top-10% mass", f"{t['top10pct_mass']*100:.1f}%",
                 "lorenz_gini.*_top_share"))
fb = PA["flippable_band"]
for fam in ("phd", "cmm"):
    vals = "/".join(f"{fb[fam][l]*100:.1f}" for l in ("0.5", "1.0", "2.0", "4.0"))
    rows.append((f"(b) inset {fam} band% λ=0.5/1/2/4", vals + " %",
                 "flippable_band"))

# (c)
df = pd.read_csv(os.path.join(HERE, "a13_panel_c_iou.csv"))
for fam in ("phd", "cmm"):
    s = df[(df.dataset == fam) & (df.pct == 20)].iloc[0]
    rows.append((f"(c) {fam} IoU@20%", f"{s.iou:.3f} "
                 f"[{s.ci_lo:.3f}, {s.ci_hi:.3f}]", "a13_panel_c_iou.csv"))

# (d)
dev, xb = PA["cross_backbone_dev"], PA["cross_backbone"]
for fam in ("phd", "cmm"):
    band = [dev[fam]["band_lam2"]*100, xb[fam]["qwen3_dev"]["band_lam2"]*100,
            xb[fam]["nemotron_dev"]["band_lam2"]*100]
    med = [dev[fam]["median_abs_d"], xb[fam]["qwen3_dev"]["median_abs_d"],
           xb[fam]["nemotron_dev"]["median_abs_d"]]
    n = [dev[fam]["n"], xb[fam]["qwen3_dev"]["n"], xb[fam]["nemotron_dev"]["n"]]
    rows.append((f"(d) {fam} band% Q2.5/Q3/Nemo",
                 "/".join(f"{b:.1f}" for b in band), "cross_backbone_dev + "
                 "cross_backbone (λ=2)"))
    rows.append((f"(d) {fam} median|d|", "/".join(f"{m:.3g}" for m in med),
                 "same nodes"))
    rows.append((f"(d) {fam} n", "/".join(str(x) for x in n), "same nodes"))

w = max(len(a) for a, _, _ in rows)
for a, b, c in rows:
    print(f"{a:<{w}}  {b:<34} {c}")
