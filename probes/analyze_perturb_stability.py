# -*- coding: utf-8 -*-
"""analyze_perturb_stability.py — zero-GPU analysis of the perturbation-stability
probe (avh_perturb_{crf32,halfres,audnoise}_shard0of1.jsonl).

Three readouts per perturbation ladder:
  R1 per-token Spearman(delta_clean, delta_perturb) per channel, pooled over
     all tokens and as mean of within-sample rhos.
  R2 sample-level: Spearman(frag_clean, frag_perturb) over samples, and the
     confidence-blindspot verdict rho(conf, frag) recomputed on perturbed
     deltas -- does the verdict drift?  (conf = base norm_logp, unchanged by
     construction; frag = sum_t sqrt(dv^2+da^2), same as
     recompute_avh_blindspot.py / unified_cf_object.)
  R3 magnitude drift: mean|delta| ratio pert/clean per channel, mean frag ratio.

Run with /usr/bin/python3.10 (numpy+scipy present).
"""
import os, json, math
import numpy as np
from scipy.stats import spearmanr

ROOT = "/home/tsq/rhq/omniconfess/OmniConfess"
BASE = os.path.join(ROOT, "outputs/cf_probe")
OUTD = "/home/tsq/rhq/omniconfess/outputs/problem_structure_fig"
SCORED = os.path.join(ROOT, "outputs/avhbench_base_model/",
                      "avhbench_base_model_full.jsonl")
sys_path = ROOT
PERTURBS = ["crf32", "halfres", "audnoise"]
rng = np.random.default_rng(20260823)


def boot_ci(a, b, B=2000):
    n = len(a); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            v = spearmanr(a[idx], b[idx]).statistic
            if v == v:
                vals.append(v)
        except Exception:
            pass
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def verdict(rho, lo, hi):
    if rho < 0 and hi < 0:
        return "CALIBRATED"
    if lo <= 0 <= hi or rho >= 0:
        return "BLIND"
    return "WEAK/PARTIAL"


def load_conf_map():
    """Rebuild the identical stratified sample order used by the probe and map
    output row id -> base norm_logp (matched sequentially, in sample order)."""

    ns = {}
    ns = {}
    exec(compile('def token_f1(pred, gt):\n    import string\n    from collections import Counter\n    STOP = set(["a","an","the","and","or","but","if","because","as","what",\n        "which","this","that","these","those","then","just","so","than",\n        "such","both","through","about","for","is","are","was","were",\n        "of","while","during","to","in","on","at","by","with","it","there"])\n    def f1t(x):\n        x = str(x).lower().translate(str.maketrans("", "", string.punctuation))\n        return [w for w in x.split() if w not in STOP and w.strip()]\n    p_t, g_t = f1t(pred), f1t(gt)\n    if not p_t or not g_t:\n        return 0.0\n    common = Counter(p_t) & Counter(g_t)\n    hit = sum(common.values())\n    p, r = hit/len(p_t), hit/len(g_t)\n    return (2*p*r)/(p+r) if (p+r) > 0 else 0.0\n\ndef stratified_sample(recs, n, seed=42):\n    import random\n    rng = random.Random(seed)\n    ordered = sorted(recs, key=lambda r: r[\'_f1\'])\n    k = len(ordered) // 3\n    tiers = [ordered[:k], ordered[k:2*k], ordered[ordered and 2*k:]]\n    for t in tiers:\n        rng.shuffle(t)\n    picked = []\n    while len(picked) < n:\n        for t in tiers:\n            if t and len(picked) < n:\n                picked.append(t.pop())\n    return picked\n', "pca_helpers", "exec"), ns)
    token_f1, stratified_sample = ns["token_f1"], ns["stratified_sample"]
    recs = [json.loads(l) for l in open(SCORED)]
    recs = [r for r in recs if r.get("token_ids")]
    for r in recs:
        r["_f1"] = token_f1(r["pred"], r["gt"])
    sample = stratified_sample(recs, 400, 42)[:100]
    return sample


def frag(dv, da):
    return float(np.sqrt(np.asarray(dv, float) ** 2 +
                         np.asarray(da, float) ** 2).sum())


def main():
    sample = load_conf_map()
    results = {"n_sample_selected": len(sample), "perturbs": {}}

    # id -> norm_logp (first occurrence in sample order; the probe preserves
    # sample order, output rows are a subsequence)
    for pert in PERTURBS:
        path = os.path.join(BASE, f"avh_perturb_{pert}_shard0of1.jsonl")
        rows = [json.loads(l) for l in open(path)]
        rows = [r for r in rows if r.get("aligned")]
        # join conf: iterate sample in order, consume rows with matching id
        conf_by_row = []
        it = iter(rows)
        cur = next(it, None)
        for rec in sample:
            if cur is not None and str(rec["id"]) == cur["id"]:
                conf_by_row.append((cur, float(rec["norm_logp"])))
                cur = next(it, None)
        # any leftover rows (shouldn't happen) -- fall back to id match
        while cur is not None:
            m = [r for r in sample if str(r["id"]) == cur["id"]]
            conf_by_row.append((cur, float(m[0]["norm_logp"]) if m else float("nan")))
            cur = next(it, None)

        # R1 per-token
        tok = {}
        per_sample_rho = {"audio": [], "video": []}
        for ch in ("audio", "video"):
            all0, all1 = [], []
            for r, _ in conf_by_row:
                d0 = np.asarray(r[f"delta_{ch}_clean"], float)
                d1 = np.asarray(r[f"delta_{ch}_{pert}"], float)
                if len(d0) >= 5:
                    rr = spearmanr(d0, d1).statistic
                    if rr == rr:
                        per_sample_rho[ch].append(rr)
                all0.append(d0); all1.append(d1)
            a = np.concatenate(all0); b = np.concatenate(all1)
            pooled = spearmanr(a, b).statistic
            tok[ch] = {
                "pooled_spearman": round(float(pooled), 4),
                "mean_within_sample_spearman": round(
                    float(np.mean(per_sample_rho[ch])), 4),
                "n_tokens": int(len(a)),
                "mean_abs_clean": round(float(np.abs(a).mean()), 4),
                "mean_abs_pert": round(float(np.abs(b).mean()), 4),
            }

        # R2 sample-level frag + verdict
        conf = np.array([c for _, c in conf_by_row], float)
        f0 = np.array([frag(r["delta_video_clean"], r["delta_audio_clean"])
                       for r, _ in conf_by_row])
        f1 = np.array([frag(r[f"delta_video_{pert}"], r[f"delta_audio_{pert}"])
                       for r, _ in conf_by_row])
        rho_ff = spearmanr(f0, f1).statistic
        rho_c0, lo0, hi0 = spearmanr(conf, f0).statistic, *boot_ci(conf, f0)
        rho_c1, lo1, hi1 = spearmanr(conf, f1).statistic, *boot_ci(conf, f1)
        samp = {
            "n": len(conf),
            "spearman_frag_clean_vs_pert": round(float(rho_ff), 4),
            "blindspot_clean": {
                "rho_conf_frag": round(float(rho_c0), 4),
                "ci95": [round(lo0, 4), round(hi0, 4)],
                "verdict": verdict(rho_c0, lo0, hi0)},
            "blindspot_perturbed": {
                "rho_conf_frag": round(float(rho_c1), 4),
                "ci95": [round(lo1, 4), round(hi1, 4)],
                "verdict": verdict(rho_c1, lo1, hi1)},
            "verdict_drift": verdict(rho_c0, lo0, hi0) != verdict(rho_c1, lo1, hi1),
        }

        # R3 magnitude drift
        mag = {
            "mean_frag_clean": round(float(f0.mean()), 4),
            "mean_frag_pert": round(float(f1.mean()), 4),
            "frag_ratio": round(float(f1.mean() / f0.mean()), 4),
            "abs_delta_audio_ratio": round(
                tok["audio"]["mean_abs_pert"] / max(tok["audio"]["mean_abs_clean"], 1e-9), 4),
            "abs_delta_video_ratio": round(
                tok["video"]["mean_abs_pert"] / max(tok["video"]["mean_abs_clean"], 1e-9), 4),
        }

        results["perturbs"][pert] = {
            "n_rows": len(rows), "n_used": len(conf_by_row),
            "r1_token_level": tok, "r2_sample_level": samp,
            "r3_magnitude": mag}
        print(pert, json.dumps(results["perturbs"][pert], indent=1), flush=True)

    # stability verdict per perturbation: pooled token spearman > 0.7 => stable
    for pert, d in results["perturbs"].items():
        rhos = [d["r1_token_level"][c]["pooled_spearman"] for c in ("audio", "video")]
        d["ruler_verdict"] = "STABLE" if min(rhos) > 0.7 else \
            ("MARGINAL" if min(rhos) > 0.5 else "SENSITIVE")

    results["method_notes"] = {
        "model": "Qwen2.5-Omni-7B thinker, bf16, teacher-forced on base greedy token_ids",
        "crf32": "libx264 CRF32 (ffmpeg 7.0.2 static), audio track dropped in the perturbed copy",
        "halfres": "ffmpeg scale to half linear resolution (even dims), loader resizes back to 224x224",
        "audnoise": "gaussian noise, SNR 20 dB over the whole clip, numpy, per-sample seeded (crc32(id))",
        "clean_reference": "recomputed in the same process as the perturbed views "
                           "(the scored jsonl was regenerated after the bigN probe, so "
                           "bigN rows' token sequences no longer match)",
        "token_alignment": "verified per sample: len(lp) == n_tokens for all 6 views, "
                           "prompt skeleton unchanged (placeholder layout identical)",
    }

    os.makedirs(OUTD, exist_ok=True)
    json.dump(results, open(os.path.join(OUTD, "perturbation_stability.json"), "w"),
              indent=2, ensure_ascii=False)

    # ---------------- markdown ----------------
    lines = ["# 输入扰动下 delta 的稳定性（测量尺子可靠性）", "",
             "问题：`delta_audio / delta_video` 这把尺子对输入的表面扰动（重压缩/降分辨率/加噪）"
             "是否稳定？扰动不改变文本与 token 序列（teacher-forced 同一批 base greedy token_ids，"
             "逐样本核对 6 个视图的 lp 长度与 prompt 骨架一致）。", "",
             "数据：avhbench，bigN 探针同款抽样（stratified_sample n=400 seed=42）前 "
             f"{results['n_sample_selected']} 条；clean 参照与扰动视图同进程重算"
             "（scored 文件在 bigN 之后被重跑，bigN 行的 token 序列已不匹配，故不直接复用其 lp）。", "",
             "| 扰动 | 逐token ρ(dA) | 逐token ρ(dV) | 样本级 ρ(frag) | 幅值比 dA | 幅值比 dV | frag比 | 尺子verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for pert, d in results["perturbs"].items():
        t, s2, m = d["r1_token_level"], d["r2_sample_level"], d["r3_magnitude"]
        lines.append(
            f"| {pert} | {t['audio']['pooled_spearman']} | {t['video']['pooled_spearman']} "
            f"| {s2['spearman_frag_clean_vs_pert']} | {m['abs_delta_audio_ratio']} "
            f"| {m['abs_delta_video_ratio']} | {m['frag_ratio']} | {d['ruler_verdict']} |")
    lines += ["", "## 盲区 verdict 漂移（ρ(conf, frag)，conf=base norm_logp 不变）", "",
              "| 扰动 | clean ρ [CI] verdict | perturbed ρ [CI] verdict | verdict 漂移 |",
              "|---|---|---|---|"]
    for pert, d in results["perturbs"].items():
        b0 = d["r2_sample_level"]["blindspot_clean"]
        b1 = d["r2_sample_level"]["blindspot_perturbed"]
        lines.append(f"| {pert} | {b0['rho_conf_frag']} {b0['ci95']} {b0['verdict']} "
                     f"| {b1['rho_conf_frag']} {b1['ci95']} {b1['verdict']} "
                     f"| {d['r2_sample_level']['verdict_drift']} |")
    lines += ["", "## 读数", ""]
    for pert, d in results["perturbs"].items():
        t = d["r1_token_level"]
        lines.append(
            f"- **{pert}**: 逐token池化 ρ(dA)={t['audio']['pooled_spearman']}，"
            f"ρ(dV)={t['video']['pooled_spearman']}（样本内均值 "
            f"{t['audio']['mean_within_sample_spearman']}/"
            f"{t['video']['mean_within_sample_spearman']}）→ 尺子{d['ruler_verdict']}。")
    stable = [d["ruler_verdict"] == "STABLE" for d in results["perturbs"].values()]
    if all(stable):
        lines.append("\n**结论：三档扰动下逐 token 相关均 >0.7，尺子稳。** δ 度量的是模态贡献的"
                     "相对结构，对表面保真度（码率/分辨率/噪底）不敏感。")
    elif not any(stable):
        lines.append("\n**结论：δ 对表面特征敏感（limitations 必写）。** 低相关说明部分 δ 信号"
                     "来自编码伪影而非模态内容贡献。")
    else:
        lines.append("\n**结论：混合。** 稳定通道与敏感通道并存，敏感方向在 limitations 中标注。")
    lines += ["", "方法与配置见 `perturbation_stability.json` 的 method_notes；"
              "逐样本数据在 `outputs/cf_probe/avh_perturb_*_shard0of1.jsonl`。"]
    open(os.path.join(OUTD, "perturbation_stability.md"), "w").write("\n".join(lines))
    print("wrote", os.path.join(OUTD, "perturbation_stability.md"))


if __name__ == "__main__":
    main()
