# -*- coding: utf-8 -*-
"""recompute_avh_blindspot.py -- recompute the Axis-3 (confidence-blindspot)
avhbench cell at FULL per-sample granularity from the enlarged CF probe.

Why this exists: unified_cf_object.py joins the avhbench probe to base rows by
the base file's integer ``id``, but that id is shard-local -- 645 base rows carry
only 81 distinct ids -- so the join collapsed the sample to n=81 (CI crossed 0,
verdict INCONCLUSIVE). The base file's UNIQUE key is (id, gt, pred) (645 unique,
0 collisions, verified). The enlarged probe (avh_probe_bigN) writes id, gt, pred
per sample, so we join on that triple, recovering per-sample norm_logp + Token-F1
without any collapse.

Fragility (identical to unified_cf_object.load_gen for avhbench):
    per token mag = sqrt(dv^2 + da^2);  sample frag_sum = sum_t mag.
Confidence = base norm_logp. Blindspot = Spearman rho(conf, frag) with a 2000x
bootstrap CI + 2000x permutation p, SAME rng seed / verdict rule as leg3.

Read-only; no GPU; nothing fabricated.
"""
import os, json, math
import numpy as np

ROOT = "/home/tsq/rhq/omniconfess/OmniConfess"
PROBE = os.path.join(ROOT, "outputs/cf_probe/avh_probe_bigN_shard0of1.jsonl")
BASE = os.path.join(ROOT, "outputs/avhbench_base_model/"
                          "avhbench_base_model_full_single_scored.jsonl")
rng = np.random.default_rng(20260820)   # same seed as leg3_blindspot_math


def _rankdata(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    sx = x[order]; i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra = _rankdata(a) - _rankdata(a).mean()
    rb = _rankdata(b) - _rankdata(b).mean()
    d = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def boot_ci_spearman(a, b, B=2000):
    n = len(a); vals = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        vals[i] = spearman(a[idx], b[idx])
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def perm_p_spearman(a, b, B=2000):
    obs = spearman(a, b)
    if obs != obs:
        return float("nan"), obs
    bc = np.asarray(b, float).copy(); cnt = 0
    for _ in range(B):
        bp = bc.copy(); rng.shuffle(bp)
        if abs(spearman(a, bp)) >= abs(obs) - 1e-12:
            cnt += 1
    return (cnt + 1) / (B + 1), obs


def main():
    # base: (id, gt, pred) -> (norm_logp, Token-F1)
    base = {}
    for l in open(BASE):
        r = json.loads(l)
        k = (str(r["id"]), r["gt"], r["pred"])
        base[k] = (r.get("norm_logp"),
                   r.get("eval_metrics", {}).get("Token-F1"))

    probe = [json.loads(l) for l in open(PROBE)]

    conf, frag, corr = [], [], []
    n_join_miss = 0
    f1s_for_med = []
    joined = []
    for p in probe:
        k = (str(p["id"]), p["gt"], p["pred"])
        b = base.get(k)
        if b is None:
            n_join_miss += 1
            continue
        nl, f1 = b
        if nl is None:
            continue
        dv = np.asarray(p["delta_video"], float)
        da = np.asarray(p["delta_audio"], float)
        mag = np.sqrt(dv * dv + da * da)
        joined.append((float(nl), float(mag.sum()),
                       f1 if f1 is not None else None))
        if f1 is not None:
            f1s_for_med.append(f1)

    med_f1 = float(np.median(f1s_for_med)) if f1s_for_med else 0.0
    for nl, fr, f1 in joined:
        conf.append(nl); frag.append(fr)
        corr.append(1 if (f1 is not None and f1 >= med_f1) else 0)

    conf = np.asarray(conf, float); frag = np.asarray(frag, float)
    corr = np.asarray(corr, int)
    n = len(conf)

    rho = spearman(conf, frag)
    lo, hi = boot_ci_spearman(conf, frag)
    pp, _ = perm_p_spearman(conf, frag)
    ci_cross0 = (lo <= 0 <= hi)
    if rho < 0 and hi < 0 and pp < 0.05:
        vd = "CALIBRATED"
    elif ci_cross0 or rho >= 0:
        vd = "BLIND"
    else:
        vd = "WEAK/PARTIAL"

    # controls
    rho_conf_corr = spearman(conf, corr) if len(set(corr.tolist())) > 1 else float("nan")
    rho_frag_corr = spearman(frag, corr) if len(set(corr.tolist())) > 1 else float("nan")

    out = {
        "cell": "avhbench blindspot (enlarged probe, (id,gt,pred) join, no id-collapse)",
        "probe_file": os.path.relpath(PROBE, ROOT),
        "n_probe_rows": len(probe),
        "n_join_miss": n_join_miss,
        "n": n,
        "frag_def": "sum_t sqrt(dv^2+da^2)",
        "conf_def": "base norm_logp",
        "rho_conf_frag": round(rho, 4),
        "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
        "ci_crosses_0": bool(ci_cross0),
        "perm_p": round(pp, 4),
        "verdict": vd,
        "f1_median": round(med_f1, 4),
        "rho_conf_correct": round(rho_conf_corr, 4),
        "rho_frag_correct": round(rho_frag_corr, 4),
        "prev_cell": {"n": 71, "rho": -0.1108, "ci": [-0.3369, 0.1139],
                      "perm_p": 0.3528, "verdict": "INCONCLUSIVE",
                      "cause": "id-collapse 645->81 + thin probe coverage"},
    }
    outp = os.path.join(ROOT, "outputs/problem_structure_fig",
                        "avh_blindspot_bigN.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print("wrote", outp)


if __name__ == "__main__":
    main()
