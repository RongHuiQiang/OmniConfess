# -*- coding: utf-8 -*-
"""compute_dev_stats.py — Qwen2.5 dev-sample cross-backbone stats (zero GPU).

Purpose: supply the Qwen2.5 point of panel (d) with a DEV sample that matches
the Qwen3 (n=100/98) and Nemotron (n=100/98) dev-probe protocol, and append a
`cross_backbone_dev` node to a13_fig2_panels.json (append-only; existing keys
are left untouched).

Dev-sample choice (investigated, see RENDER_NOTES.md):
  * dev100_ids.json contains phd.json uid suffixes (dataset row indices
    0..905). The probe files' `id` fields live in DIFFERENT id spaces
    (phd probes: scored-file group ids 0..113, only 17 unique overlap with
    dev100_ids; cmm probes: shard-local ids). Filtering the full probes by
    dev100_ids therefore does NOT yield a dev100 subsample (136 rows / 17
    questions for phd, 104 rows / 13 questions for cmm).
  * The protocol-matched Qwen2.5 dev probes already exist and were produced by
    the same stratified (gt_yes x base_correct, 25/cell, seed 42) scripts as
    the Qwen3/Nemotron dev probes:
        outputs/cf_probe/phd_probe.jsonl  (n=100, 25/25/25/25)
        outputs/cf_probe/cmm_probe.jsonl  (n=87, 25/25/24/13)
    a3_dose_response.py already cites phd_probe.jsonl as "dev100".
  => primary dev stats use those files; the dev100_ids-intersection subset is
     computed as well and stored under `cross_backbone_dev_alternative_dev100ids`
     for transparency (rejected: wrong id space, not a 100-sample dev set).

Stats per family (same conventions as a13_fig2_panels.py):
  n, median_abs_d, band_lam2 = mean(|margin_full| < 2*|d|), bootstrap 95% CI
  (1000 resamples, seed 42, row-joint resampling for the band).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CF = os.path.join(ROOT, "outputs", "cf_probe")

rng = np.random.default_rng(42)
N_BOOT = 1000


def jl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def stats(rows, dkey):
    d = np.abs(np.array([r[dkey] for r in rows], float))
    m = np.abs(np.array([r["margin_full"] for r in rows], float))
    band = float(np.mean(m < 2 * d))
    boots = []
    for _ in range(N_BOOT):
        ii = rng.integers(0, len(d), len(d))
        boots.append(np.mean(m[ii] < 2 * d[ii]))
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return {"n": len(rows), "median_abs_d": float(np.median(d)),
            "band_lam2": band, "band_lam2_ci95": ci,
            "source": os.path.basename(dkey)}


def main():
    phd_dev = jl(os.path.join(CF, "phd_probe.jsonl"))
    cmm_dev = jl(os.path.join(CF, "cmm_probe.jsonl"))
    phd_full = jl(os.path.join(CF, "phd_probe_full.jsonl"))
    cmm_full = jl(os.path.join(CF, "cmm_probe_full.jsonl"))
    dev_ids = set(json.load(open(os.path.join(CF, "dev100_ids.json"))))

    node = {
        "_note": ("Qwen2.5 dev point for panel (d). Primary = existing "
                  "protocol-matched stratified dev probes (phd_probe.jsonl "
                  "n=100, cmm_probe.jsonl n=87; same 25/cell seed-42 protocol "
                  "as the Qwen3/Nemotron dev probes). dev100_ids.json indexes "
                  "phd.json uid suffixes and does NOT match the probe id "
                  "spaces, so filtering the full probes by it yields 136 rows "
                  "/17 questions (phd) and 104 rows/13 questions (cmm) — "
                  "kept as rejected alternative below."),
        "phd": stats(phd_dev, "delta_noctx"),
        "cmm": stats(cmm_dev, "d_cf_sum"),
        "alternative_dev100ids_intersection_rejected": {
            "phd": stats([r for r in phd_full if r["id"] in dev_ids],
                         "delta_noctx"),
            "cmm": stats([r for r in cmm_full if r["id"] in dev_ids],
                         "d_cf_sum"),
        },
    }

    path = os.path.join(HERE, "a13_fig2_panels.json")
    data = json.load(open(path))
    data["panel_a_concentration"]["cross_backbone_dev"] = node  # append
    with open(path, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    print(json.dumps(node, indent=1))


if __name__ == "__main__":
    main()
