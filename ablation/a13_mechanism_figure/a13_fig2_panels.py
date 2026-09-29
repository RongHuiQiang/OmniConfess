# -*- coding: utf-8 -*-
"""a13_fig2_panels.py — mechanism-figure statistics (offline, zero GPU).

Recomputes all numbers for the paper's "problem-mechanism" three-panel figure
(mirror of the decision-token paper's Fig 2):

  (a) pathology concentration  — Lorenz/Gini of |counterfactual dependence|,
      flippable-band share, cross-backbone collapse of both.
  (b) channel localization     — cmm 2x2 |d_cf_audio|/|d_cf_video| by lesion
      group + enrichment ratio; avh delta_audio/delta_video; phd text-vs-vision
      channel medians with qonly prior-strength reference; channel x dataset
      dependence-mass matrix.
  (c) profile validity         — IoU(top-p% |d| set, top-p% |margin_full| set)
      with bootstrap CI; AUROC table (a3 citation + cross-backbone collapse
      computed here); entropy-vs-grounding feature race from bon_probe.

Outputs (this directory):
  a13_fig2_panels.json          — all statistics
  a13_panel_a_lorenz.csv        — Lorenz curve points (all conditions)
  a13_panel_b_channel_matrix.csv— channel x dataset median matrix
  a13_panel_c_iou.csv           — IoU table

Field notes / deviations from the task sheet (also recorded in JSON.meta):
  - cmm_probe_full.jsonl has 720 rows, not 721 (91+90*7=720; w5 has 89).
  - avh probes have 120 rows total (4 x 30), not ~130.
  - cmm ids are shard-local -> everything handled by file line position; phd ids
    are also non-unique (114 unique / 906) so phd_probe_qonly.jsonl CANNOT be
    joined to phd_probe_full; it is used only as a standalone distribution.
  - a3_dose_response.json does not contain the Nemotron probe AUCs, so
    phd@Nemotron and cmm@Nemotron AUCs are computed here from the probe files.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CF = os.path.join(ROOT, "outputs", "cf_probe")

rng = np.random.default_rng(42)
N_BOOT, SEED = 1000, 42


# ---------------------------------------------------------------- utilities
def jl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def gini(vals):
    """Gini coefficient of nonnegative mass vector."""
    x = np.sort(np.abs(np.asarray(vals, dtype=float)))
    if len(x) < 2 or x.sum() == 0:
        return float("nan")
    n = len(x)
    cum = np.cumsum(x)
    return float(1.0 - 2.0 * np.sum(cum) / (n * cum[-1]) + 1.0 / n)


def lorenz_points(vals, n_pts=101):
    x = np.sort(np.abs(np.asarray(vals, dtype=float)))
    total = x.sum()
    qs = np.linspace(0, 1, n_pts)
    if total == 0:
        return qs, qs
    cum = np.concatenate([[0.0], np.cumsum(x)]) / total
    pop = np.linspace(0, 1, len(x) + 1)
    return qs, np.interp(qs, pop, cum)


def boot_ci(stat_fn, data, n=N_BOOT):
    """Bootstrap 95% CI for a statistic that takes one 1-D array."""
    data = np.asarray(data, dtype=float)
    out = np.array([stat_fn(data[rng.integers(0, len(data), len(data))])
                    for _ in range(n)])
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


def boot_ci_pair(f, a, b, n=N_BOOT):
    """Bootstrap CI for a paired statistic f(a, b): resample ROW indices jointly."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b)
    out = []
    for _ in range(n):
        ii = rng.integers(0, len(a), len(a))
        v = f(a[ii], b[ii])
        if np.isfinite(v):
            out.append(v)
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


def auroc(scores, labels):
    """AUROC of score predicting label==1."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import mannwhitneyu
    u, _ = mannwhitneyu(pos, neg, alternative="greater")
    return float(u / (len(pos) * len(neg)))


def topset_iou(x, y, p):
    """IoU of top-p% |x| indices and top-p% |y| indices."""
    x, y = np.abs(np.asarray(x, float)), np.abs(np.asarray(y, float))
    n = len(x)
    k = max(1, int(round(n * p / 100.0)))
    sx = set(np.argsort(-x)[:k].tolist())
    sy = set(np.argsort(-y)[:k].tolist())
    return float(len(sx & sy) / len(sx | sy))


def median(v):
    v = np.abs(np.asarray(v, dtype=float))
    return float(np.median(v)) if len(v) else float("nan")


# ---------------------------------------------------------------- load data
phd = jl(os.path.join(CF, "phd_probe_full.jsonl"))            # 906
cmm = jl(os.path.join(CF, "cmm_probe_full.jsonl"))            # 720
qonly = jl(os.path.join(CF, "phd_probe_qonly.jsonl"))         # 906
phd_n = jl(os.path.join(CF, "phd_probe_nemotron.jsonl"))      # 100
phd_3 = jl(os.path.join(CF, "phd_probe_qwen3.jsonl"))         # 100
cmm_n = jl(os.path.join(CF, "cmm_probe_nemotron.jsonl"))      # 100
cmm_3 = jl(os.path.join(CF, "cmm_probe_qwen3.jsonl"))         # 98
avh = sum([jl(os.path.join(CF, f"avh_probe_shard{i}of4.jsonl")) for i in range(4)], [])
a3 = json.load(open(os.path.join(ROOT, "ablation", "a3_dose_response.json")))

result = {"meta": {
    "n": {"phd_full": len(phd), "cmm_full": len(cmm), "qonly": len(qonly),
          "phd_nemotron_dev": len(phd_n), "phd_qwen3_dev": len(phd_3),
          "cmm_nemotron_dev": len(cmm_n), "cmm_qwen3_dev": len(cmm_3),
          "avh": len(avh)},
    "notes": [
        "cmm_probe_full.jsonl has 720 rows (task sheet said 721); w5 shard has 89.",
        "avh probes total 120 rows (4 shards x 30), not ~130.",
        "phd ids are non-unique (114 unique / 906): phd_probe_qonly.jsonl used as "
        "standalone distribution only (no join); cmm handled by line position.",
        "Nemotron probe AUCs absent from a3_dose_response.json -> computed here.",
        "Cross-backbone rows use dev probes (n=100/98) vs Qwen2.5 full (906/720); "
        "dev-vs-full caveats apply.",
        "bootstrap: 1000 resamples, seed=42.",
    ],
}}

LAMBDAS = [0.5, 1.0, 2.0, 4.0]

# ============================================================== PANEL (a)
lorenz_rows = []  # csv rows: condition, pop_frac, mass_frac
panel_a = {"lorenz_gini": {}, "flippable_band": {}, "cross_backbone": {}}


def lorenz_block(name, vals):
    g = gini(vals)
    ci = boot_ci(gini, vals)
    qs, ms = lorenz_points(vals)
    panel_a["lorenz_gini"][name] = {"n": len(vals), "gini": g, "ci95": ci}
    for q, m in zip(qs, ms):
        lorenz_rows.append({"condition": name, "pop_frac": round(q, 4),
                            "mass_frac": round(m, 4)})
    return g


def band(vals_d, vals_m, lam):
    d = np.abs(np.asarray(vals_d, float))
    m = np.abs(np.asarray(vals_m, float))
    return float(np.mean(np.abs(m) < lam * d)) if len(d) else float("nan")


phd_d_nc = [r["delta_noctx"] for r in phd]
phd_d_ni = [r["delta_noimg"] for r in phd]
phd_m = [r["margin_full"] for r in phd]
cmm_d = [r["d_cf_sum"] for r in cmm]
cmm_m = [r["margin_full"] for r in cmm]

lorenz_block("phd_q2.5_full_|delta_noctx|", phd_d_nc)
lorenz_block("phd_q2.5_full_|delta_noimg|", phd_d_ni)
lorenz_block("cmm_q2.5_full_|d_cf_sum|", cmm_d)
for gname in ("overrely_audio_ignore_visual", "overrely_visual_ignore_audio"):
    sub = [r["d_cf_sum"] for r in cmm if r["group"] == gname]
    lorenz_block(f"cmm_q2.5_full_{gname}_|d_cf_sum|", sub)

# top-x% mass share (sparsity headline: mass captured by top 10%/20% samples)
for name, vals in [("phd_|delta_noctx|", phd_d_nc), ("cmm_|d_cf_sum|", cmm_d)]:
    x = np.sort(np.abs(np.asarray(vals, float)))[::-1]
    cs = np.cumsum(x) / x.sum()
    panel_a["lorenz_gini"][name.replace("|", "") + "_top_share"] = {
        "top10pct_mass": float(cs[max(0, int(len(x) * .1) - 1)]),
        "top20pct_mass": float(cs[max(0, int(len(x) * .2) - 1)])}

panel_a["flippable_band"]["phd"] = {
    str(l): band(phd_d_nc, phd_m, l) for l in LAMBDAS}
panel_a["flippable_band"]["phd_ci"] = {
    str(l): boot_ci_pair(lambda d, m: np.mean(np.abs(m) < l * np.abs(d)),
                         phd_d_nc, phd_m) for l in LAMBDAS}
panel_a["flippable_band"]["cmm"] = {
    str(l): band(cmm_d, cmm_m, l) for l in LAMBDAS}
panel_a["flippable_band"]["cmm_ci"] = {
    str(l): boot_ci_pair(lambda d, m: np.mean(np.abs(m) < l * np.abs(d)),
                         cmm_d, cmm_m) for l in LAMBDAS}

# cross-backbone collapse (Gini + band@lam2)
def xb(name, dvals, mvals):
    d = np.asarray(dvals, float)
    m = np.abs(np.asarray(mvals, float))
    g = gini(d)
    return {"n": len(d), "gini": g, "gini_ci95": boot_ci(gini, d),
            "median_abs_d": float(np.median(np.abs(d))),
            "band_lam2": float(np.mean(m < 2 * np.abs(d))),
            "band_lam2_ci95": boot_ci_pair(
                lambda dd, mm: np.mean(np.abs(mm) < 2 * np.abs(dd)), d, m)}


panel_a["cross_backbone"]["phd"] = {
    "qwen2.5_full": xb("phd", phd_d_nc, phd_m),
    "qwen3_dev": xb("phd", [r["delta_noctx"] for r in phd_3],
                    [r["margin_full"] for r in phd_3]),
    "nemotron_dev": xb("phd", [r["delta_noctx"] for r in phd_n],
                       [r["margin_full"] for r in phd_n]),
}
panel_a["cross_backbone"]["cmm"] = {
    "qwen2.5_full": xb("cmm", cmm_d, cmm_m),
    "qwen3_dev": xb("cmm", [r["d_cf_sum"] for r in cmm_3],
                    [r["margin_full"] for r in cmm_3]),
    "nemotron_dev": xb("cmm", [r["d_cf_sum"] for r in cmm_n],
                       [r["margin_full"] for r in cmm_n]),
}

# ============================================================== PANEL (b)
panel_b = {"cmm_2x2": {}, "enrichment": {}, "avh": {}, "phd_channels": {},
           "channel_matrix": {}}

ga = "overrely_audio_ignore_visual"
gv = "overrely_visual_ignore_audio"
A = [r for r in cmm if r["group"] == ga]   # audio-lesion group
V = [r for r in cmm if r["group"] == gv]   # visual-lesion group
med = lambda rows, k: median([r[k] for r in rows])
cell = {
    "audio_group_|d_cf_audio|": med(A, "d_cf_audio"),
    "audio_group_|d_cf_video|": med(A, "d_cf_video"),
    "visual_group_|d_cf_audio|": med(V, "d_cf_audio"),
    "visual_group_|d_cf_video|": med(V, "d_cf_video"),
}
panel_b["cmm_2x2"] = cell
# enrichment: same-channel mass in lesion group vs the other group
ea = cell["audio_group_|d_cf_audio|"] / cell["visual_group_|d_cf_audio|"]
ev = cell["visual_group_|d_cf_video|"] / cell["audio_group_|d_cf_video|"]
panel_b["enrichment"] = {
    "audio_channel_in_audio_group_vs_visual_group": ea,
    "video_channel_in_visual_group_vs_audio_group": ev,
}
# two-sample bootstrap (groups have different n): resample within each group
aau = np.abs(np.array([r["d_cf_audio"] for r in A], float))
avv = np.abs(np.array([r["d_cf_audio"] for r in V], float))
vvu = np.abs(np.array([r["d_cf_video"] for r in V], float))
vaa = np.abs(np.array([r["d_cf_video"] for r in A], float))


def two_samp(x, y, n=N_BOOT):
    out = []
    for _ in range(n):
        xx = x[rng.integers(0, len(x), len(x))]
        yy = y[rng.integers(0, len(y), len(y))]
        r = np.median(xx) / np.median(yy)
        if np.isfinite(r):
            out.append(r)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


panel_b["enrichment"]["audio_ci95"] = two_samp(aau, avv)
panel_b["enrichment"]["video_ci95"] = two_samp(vvu, vaa)

# delta_audio/delta_video are per-token lists (token-level counterfactual logit
# shifts); use flattened token-level stats plus per-sample medians.
da_tok = np.array([t for r in avh for t in r["delta_audio"]], float)
dv_tok = np.array([t for r in avh for t in r["delta_video"]], float)
da = np.array([np.median(np.abs(r["delta_audio"])) for r in avh], float)
dv = np.array([np.median(np.abs(r["delta_video"])) for r in avh], float)
f1 = np.array([r["f1"] for r in avh], float)
wrong = f1 < 0.5  # low-F1 (description) split as "wrong-ish" stratum
panel_b["avh"] = {
    "n": len(avh), "n_tokens": int(len(da_tok)),
    "note": "delta_* are per-token lists; median/mean/frac_negative are "
            "token-level over all 120 samples; strata use per-sample median|d|",
    "delta_audio": {"median": float(np.median(da_tok)),
                    "mean": float(da_tok.mean()),
                    "frac_negative": float(np.mean(da_tok < 0))},
    "delta_video": {"median": float(np.median(dv_tok)),
                    "mean": float(dv_tok.mean()),
                    "frac_negative": float(np.mean(dv_tok < 0))},
    "low_f1_lt0.5": {
        "n": int(wrong.sum()),
        "median_|delta_audio|": float(np.median(np.abs(da[wrong]))),
        "median_|delta_video|": float(np.median(np.abs(dv[wrong])))},
    "high_f1_ge0.5": {
        "n": int((~wrong).sum()),
        "median_|delta_audio|": float(np.median(np.abs(da[~wrong]))),
        "median_|delta_video|": float(np.median(np.abs(dv[~wrong])))},
}

panel_b["phd_channels"] = {
    "median_|delta_noctx|_text_prior_channel": median(phd_d_nc),
    "median_|delta_noimg|_vision_channel": median(phd_d_ni),
    "ratio_noctx_over_noimg": median(phd_d_nc) / median(phd_d_ni),
    "median_|margin_qonly|_question_only_prior": median(
        [r["margin_qonly"] for r in qonly]),
    "ci95_noctx": boot_ci(lambda v: np.median(np.abs(v)), phd_d_nc),
    "ci95_noimg": boot_ci(lambda v: np.median(np.abs(v)), phd_d_ni),
}

# channel x dataset dependence-mass matrix (median |d| per channel)
chan_rows = []
matrix = {
    "phd(q2.5,906)": {"text_prior(noctx)": median(phd_d_nc),
                      "vision(noimg)": median(phd_d_ni)},
    "cmm(q2.5,720)": {"audio": med(cmm, "d_cf_audio"),
                      "video": med(cmm, "d_cf_video")},
    "avh(q2.5,120)": {"audio": float(np.median(np.abs(da))),
                      "video": float(np.median(np.abs(dv)))},
    "phd@qwen3(dev100)": {"text_prior(noctx)": median(
        [r["delta_noctx"] for r in phd_3])},
    "phd@nemotron(dev100)": {"text_prior(noctx)": median(
        [r["delta_noctx"] for r in phd_n])},
}
panel_b["channel_matrix"] = matrix
for ds, chans in matrix.items():
    for ch, v in chans.items():
        chan_rows.append({"dataset": ds, "channel": ch,
                          "median_abs_d": round(v, 4)})

# ============================================================== PANEL (c)
panel_c = {"iou": {}, "auroc_cited_from_a3": a3["probe_aucs"],
           "cross_backbone_auc": {}, "entropy_race": {}}

PCT = [1, 5, 10, 20]
for p in PCT:
    v = topset_iou(phd_d_nc, phd_m, p)
    ci = [topset_iou(phd_d_nc[rng.permutation(len(phd))], phd_m, p) if False
          else float("nan")]  # replaced below
    # proper bootstrap: resample rows jointly
    idx = np.arange(len(phd))
    boots = []
    darr = np.abs(np.array(phd_d_nc, float))
    marr = np.abs(np.array(phd_m, float))
    for _ in range(N_BOOT):
        ii = rng.integers(0, len(phd), len(phd))
        boots.append(topset_iou(darr[ii], marr[ii], p))
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    panel_c["iou"][f"phd_top{p}pct_|delta_noctx|_vs_|margin_full|"] = {
        "iou": v, "ci95": ci}

darr_c = np.abs(np.array(cmm_d, float))
marr_c = np.abs(np.array(cmm_m, float))
for p in PCT:
    v = topset_iou(darr_c, marr_c, p)
    boots = []
    for _ in range(N_BOOT):
        ii = rng.integers(0, len(cmm), len(cmm))
        boots.append(topset_iou(darr_c[ii], marr_c[ii], p))
    panel_c["iou"][f"cmm_top{p}pct_|d_cf_sum|_vs_|margin_full|"] = {
        "iou": v, "ci95": [float(np.percentile(boots, 2.5)),
                           float(np.percentile(boots, 97.5))]}

# cross-backbone AUROC collapse — same signal/label convention as
# ablation/a3_dose_response.py auc_block(): signal = lp_yes counterfactual
# shift, label = gt_yes. (Using base_correct as label gives different numbers;
# a3's published values use gt_yes, so we match that.)
def d_sig(r):
    return r["lp_yes_noctx"] - r["lp_yes_full"]


panel_c["cross_backbone_auc"]["phd_d_cf_noctx"] = {
    "qwen2.5_dev100(cited a3)": 0.767,
    "qwen2.5_full906(cited a3)": 0.68,
    "qwen3_dev100": auroc([d_sig(r) for r in phd_3],
                          [r["gt_yes"] for r in phd_3]),
    "nemotron_dev100": auroc([d_sig(r) for r in phd_n],
                             [r["gt_yes"] for r in phd_n]),
}
panel_c["cross_backbone_auc"]["cmm_neg_d_cf_sum"] = {
    "qwen2.5_full720(cited a3)": 0.595,
    "qwen3_dev98": auroc([-(r["d_cf_audio"] + r["d_cf_video"]) for r in cmm_3],
                         [r["gt_yes"] for r in cmm_3]),
    "nemotron_dev100": auroc([-(r["d_cf_audio"] + r["d_cf_video"])
                              for r in cmm_n],
                             [r["gt_yes"] for r in cmm_n]),
}
# small-sample CI for the dev AUCs (bootstrap over rows)
def auc_ci(rows, score):
    s = np.array([score(r) for r in rows], float)
    y = np.array([r["gt_yes"] for r in rows], int)
    boots = []
    for _ in range(N_BOOT):
        ii = rng.integers(0, len(s), len(s))
        if len(set(y[ii])) < 2:
            continue
        boots.append(auroc(s[ii], y[ii]))
    return [float(np.nanpercentile(boots, 2.5)),
            float(np.nanpercentile(boots, 97.5))]


panel_c["cross_backbone_auc"]["phd_d_cf_noctx_ci95"] = {
    "qwen3_dev100": auc_ci(phd_3, d_sig),
    "nemotron_dev100": auc_ci(phd_n, d_sig)}
panel_c["cross_backbone_auc"]["cmm_neg_d_cf_sum_ci95"] = {
    "qwen3_dev98": auc_ci(cmm_3, lambda r: -(r["d_cf_audio"] + r["d_cf_video"])),
    "nemotron_dev100": auc_ci(
        cmm_n, lambda r: -(r["d_cf_audio"] + r["d_cf_video"]))}

# entropy-vs-grounding race from bon_probe (within-pool AUC, median-F1 split)
def bon_race(ds):
    d = json.load(open(os.path.join(ROOT, "outputs", "bon_probe",
                                    f"{ds}_bon_feat.json")))
    feats = {"CE_top5mean": [], "CE_max": [], "g_content(delta)": [],
             "g_mean_claim(delta)": []}
    labs = {k: [] for k in feats}
    for r in d["rows"]:
        f1s = sorted(c["f1"] for c in r["candidates"])
        med_f1 = f1s[len(f1s) // 2]
        for c in r["candidates"]:
            vals = {"CE_top5mean": c["CE"]["CE_top5mean"],
                    "CE_max": c["CE"]["CE_max"],
                    "g_content(delta)": c.get("g_content"),
                    "g_mean_claim(delta)": c.get("g_mean_claim")}
            good = c["f1"] >= med_f1
            for k, v in vals.items():
                if v is None:
                    continue
                feats[k].append(-v if k.startswith("CE") else v)  # lower CE = better
                labs[k].append(1 if good else 0)
    return {k: {"auc": auroc(feats[k], labs[k]), "n": len(feats[k]),
                "note": "CE features sign-flipped (low entropy = confident); "
                        "good = per-sample F1 >= median"}
            for k in feats}


panel_c["entropy_race"] = {"ragtruth": bon_race("ragtruth"),
                           "haloquest": bon_race("haloquest")}

# ---------------------------------------------------------------- write out
result["panel_a_concentration"] = panel_a
result["panel_b_channel"] = panel_b
result["panel_c_validity"] = panel_c

with open(os.path.join(HERE, "a13_fig2_panels.json"), "w") as f:
    json.dump(result, f, indent=1, ensure_ascii=False)

import csv
with open(os.path.join(HERE, "a13_panel_a_lorenz.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["condition", "pop_frac", "mass_frac"])
    w.writeheader()
    w.writerows(lorenz_rows)
with open(os.path.join(HERE, "a13_panel_b_channel_matrix.csv"), "w",
          newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "channel", "median_abs_d"])
    w.writeheader()
    w.writerows(chan_rows)
iou_rows = [{"dataset": k.split("_")[0], "pct": p, "iou": v["iou"],
             "ci_lo": v["ci95"][0], "ci_hi": v["ci95"][1]}
            for p in PCT
            for k, v in panel_c["iou"].items() if f"top{p}pct" in k]
with open(os.path.join(HERE, "a13_panel_c_iou.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "pct", "iou",
                                      "ci_lo", "ci_hi"])
    w.writeheader()
    w.writerows(iou_rows)

print(json.dumps({"a_gini": panel_a["lorenz_gini"],
                  "a_band_phd": panel_a["flippable_band"]["phd"],
                  "a_band_cmm": panel_a["flippable_band"]["cmm"],
                  "a_xb": panel_a["cross_backbone"],
                  "b_enrich": panel_b["enrichment"],
                  "b_phd": panel_b["phd_channels"],
                  "b_matrix": matrix,
                  "b_avh": panel_b["avh"],
                  "c_iou": {k: round(v["iou"], 3) for k, v in
                            panel_c["iou"].items()},
                  "c_xb_auc": panel_c["cross_backbone_auc"],
                  "c_entropy": panel_c["entropy_race"]},
             indent=1))
