#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_localizer_v2.py — 离线诊断: 旧 rho_i span vs 新 delta_img_tok 定位结果
对比 Round 7 四类失败样本。不需要 GPU。

运行:
    cd /home/tsq/rhq/omniconfess/OmniConfess
    python3 scripts/diagnose_localizer_v2.py 2>&1 | head -200
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import freeform_repair as fr
import localizer_v2 as lv2

# ── target sample ids per category ───────────────────────────────────────────
# Cat1: 正确拒答 (base_f1>=60, has refusal words, delta<0) — 旧方法错误编辑了
CAT1_IDS = ["601", "179"]
# Cat3: 部分正确 (30<=base_f1<70, n_spans>0, no refusal) — 旧方法乱改
CAT3_IDS = ["167", "37"]
# Cat4: 错误拒答 (has refusal, base_f1<30)
CAT4_IDS = ["608"]

ALL_TARGET_IDS = set(CAT1_IDS + CAT3_IDS + CAT4_IDS)

CACHE_PATH = os.path.join(REPO, "outputs", "cf_probe", "haloquest_textcf_tokens.jsonl")
DIAG_PATH  = os.path.join(REPO, "outputs", "freeform_grid", "round7", "diagnostics",
                           "perarm_TAU10.jsonl")


# ── helpers ───────────────────────────────────────────────────────────────────

def load_cache_by_id(path: str, ids: set) -> dict:
    """Load selected records from the raw cache jsonl (keyed by str id)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = str(r.get("id", ""))
            if rid in ids:
                out[rid] = r
    return out


def load_diag_by_id(path: str, ids: set) -> dict:
    """Load TAU10 per-arm diagnostic records."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = str(r.get("id", ""))
            if rid in ids:
                out[rid] = r
    return out


def rho_spans_for_sample(s: dict, tau: float = 1.0) -> list:
    """
    Compute rho_i with the given tau and return the selected spans.
    s is the dict from fr.load_haloquest() (contains kappa, lp_noR, modality, tokens).
    """
    cfg = fr.RepairConfig(
        tau=tau,
        prior_baseline="none",   # avoid vocab-size dependency in offline mode
        dataset_name="haloquest",
    )
    scored = fr.score_sample(s, cfg)
    return scored["spans"], scored["rho"]


def span_text(tokens: list, span: tuple) -> str:
    return "".join(tokens[span[0]:span[1]])


def sep(title: str = "") -> None:
    print()
    print("=" * 72)
    if title:
        print(f"  {title}")
        print("=" * 72)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("[load] cache...", flush=True)
    raw_cache = load_cache_by_id(CACHE_PATH, ALL_TARGET_IDS)
    print(f"[load] found {len(raw_cache)} / {len(ALL_TARGET_IDS)} target records in cache")

    print("[load] fr.load_haloquest()...", flush=True)
    all_halo = fr.load_haloquest()
    halo_by_id = {str(s["id"]): s for s in all_halo}

    print("[load] TAU10 diagnostics...", flush=True)
    diag_by_id = load_diag_by_id(DIAG_PATH, ALL_TARGET_IDS)

    CATEGORIES = [
        ("Cat1: 正确拒答 — 被旧方法错误编辑", CAT1_IDS),
        ("Cat3: 部分正确 — 旧方法乱改", CAT3_IDS),
        ("Cat4: 错误拒答（可选）", CAT4_IDS),
    ]

    for cat_title, id_list in CATEGORIES:
        sep(cat_title)
        for sid in id_list:
            raw = raw_cache.get(sid)
            s   = halo_by_id.get(sid)
            diag = diag_by_id.get(sid)
            if raw is None or s is None or diag is None:
                print(f"  [SKIP id={sid}] missing data (raw={raw is not None}, "
                      f"s={s is not None}, diag={diag is not None})")
                continue

            tokens = raw["tokens"]
            base_pred = "".join(tokens)
            question  = raw.get("question", "")
            delta_img = np.asarray(raw["delta_img_tok"], dtype=float)

            print(f"\n--- id={sid} ---")
            print(f"question  : {question}")
            print(f"base_pred : {base_pred}")
            print(f"gt        : {diag.get('gt','')[:120]}")
            print(f"base_f1   : {diag.get('base_f1_recompute', '?')}")
            print(f"delta     : {diag.get('delta', '?')}")
            print(f"repaired  : {diag.get('repaired_pred','')[:120]}")

            # ── 旧 rho_i spans (tau=1.0, same as TAU10 arm) ──────────────────
            print()
            print("[旧 rho_i] tau=1.0 spans:")
            old_spans, rho_arr = rho_spans_for_sample(s, tau=1.0)
            if not old_spans:
                print("  (none)")
            for sp in old_spans:
                txt = span_text(tokens, sp)
                mean_rho = float(rho_arr[sp[0]:sp[1]].mean())
                print(f"  tokens[{sp[0]}:{sp[1]}] = {txt!r:40s}  mean_rho={mean_rho:.3f}")

            # ── per-token delta_img_tok ───────────────────────────────────────
            print()
            print("[delta_img_tok / visual_dependence] per token:")
            top_idx = int(np.argmax(delta_img))
            for i, (tok, val) in enumerate(zip(tokens, delta_img)):
                marker = " <-- MAX" if i == top_idx else ""
                print(f"  [{i:2d}] {tok!r:20s}  delta_img={val:+.4f}{marker}")

            # ── 新 localizer_v2 candidates ───────────────────────────────────
            print()
            print("[localizer_v2] candidates (Step1-3, offline):")
            lv2_result = lv2.localize(raw)
            if not lv2_result.candidates:
                print("  (no candidates — all delta_img_tok <= min_vis_dep=0.05)")
            for c in lv2_result.candidates:
                print(f"  span_text       : {c.span_text!r}")
                print(f"  span_kind       : {c.span_kind}")
                print(f"  visual_dep(mean): {c.visual_dependence:.4f}")
                print(f"  tok_vis_dep     : {[round(v,3) for v in c.tok_visual_dependence]}")
                print(f"  is_refusal      : {c.is_refusal}")
                print(f"  decision        : {c.decision}")
                print(f"  Step4 claim     : {c.claim}")
                print()

            # ── 比较分析 ──────────────────────────────────────────────────────
            print("[分析]")
            _analyse(sid, cat_title, tokens, old_spans, rho_arr,
                     lv2_result, diag, question)

    sep("统计: 每样本 Step4 候选数 (预期 GPU 推理次数)")
    print()
    print(f"{'id':>6}  {'n_candidates':>12}  {'n_revision':>10}  {'note'}")
    for sid in sorted(ALL_TARGET_IDS):
        raw = raw_cache.get(sid)
        if raw is None:
            continue
        lr = lv2.localize(raw)
        note = ""
        if not lr.candidates:
            note = "全部低于 min_vis_dep — 无需 Step4"
        elif not lr.revision_candidates:
            note = "候选存在但 decision=PRESERVE (初始) — 真实运行需 Step4 填充"
        else:
            note = f"{len(lr.revision_candidates)} 个 REVISION_CANDIDATE"
        print(f"{sid:>6}  {len(lr.candidates):>12}  {len(lr.revision_candidates):>10}  {note}")

    print()
    print("[完成] 所有离线诊断输出完毕。Step4 (ev.verify_candidates) 需要 GPU，此处跳过。")


# ── per-sample narrative ──────────────────────────────────────────────────────

def _analyse(sid, cat, tokens, old_spans, rho_arr, lv2_result, diag, question):
    base_f1 = diag.get("base_f1_recompute", 0)
    delta   = diag.get("delta", 0)

    if sid in ("601", "179"):
        # Cat1: correct refusal, wrongly edited
        print(f"  旧方法: 选了 {len(old_spans)} 个 span (rho>1.0)，覆盖了拒答核心词。")
        print(f"  rho 信号针对的是'不确定性'而非错误 -- 拒答本身是正确的 (base_f1={base_f1})。")
        print(f"  结果: repaired 乱拼接，delta={delta}。")
        n_cand = len(lv2_result.candidates)
        refusal_cands = [c for c in lv2_result.candidates if c.is_refusal]
        print(f"  新方法: {n_cand} 个候选, 其中 is_refusal={len(refusal_cands)} 个。")
        if refusal_cands:
            print(f"  negation_refusal 类 span 在 Step5 预期 decision=PRESERVE (拒答合法)。")
            print(f"  Step4 claim 预期结果: 图中确实无法判断 → INSUFFICIENT_EVIDENCE / SUPPORTED，")
            print(f"  → decision=PRESERVE → spans=[] → 不编辑 → base_f1 保留。")
        else:
            print(f"  无 is_refusal 候选，说明拒答词 delta_img_tok 低 (视觉无关)，")
            print(f"  新方法同样不会触发编辑，与预期一致。")

    elif sid in ("167", "37"):
        # Cat3: partial correct, old method made it worse
        print(f"  旧方法: 选了 {len(old_spans)} 个 span，覆盖了部分内容词。")
        print(f"  base_f1={base_f1} 本来中等，编辑后 delta={delta} 变差。")
        n_cand = len(lv2_result.candidates)
        print(f"  新方法: {n_cand} 个候选。")
        if sid == "37":
            # "The item that is blue in this image is the beanie"
            # GT: "There is no blue item..."
            print(f"  预测: 'the beanie' 是具体断言，delta_img_tok 应较高 (颜色词依赖视觉)。")
            print(f"  Step4 claim: 'is the following statement supported: the beanie'?")
            print(f"  预期结果: 图中 beanie 未必是蓝色或不存在 → UNSUPPORTED。")
            print(f"  → decision=REVISION_CANDIDATE → 触发重生成 → 有机会修正为拒答。")
        elif sid == "167":
            print(f"  预测: 'a designer' 是无法从图中验证的具体断言。")
            print(f"  Step4 claim 预期: 图中无法确认创造者 → UNSUPPORTED。")
            print(f"  → decision=REVISION_CANDIDATE → 比旧方法定位更精准。")

    elif sid == "608":
        # Cat4: wrong refusal (base says "not named" but GT also says "we don't know")
        print(f"  base_pred 含拒答词但 base_f1={base_f1} 偏低 (GT 措辞不匹配)。")
        print(f"  旧方法选了 {len(old_spans)} 个 span，delta={delta}。")
        print(f"  新方法 {len(lv2_result.candidates)} 个候选。")
        print(f"  该样本属于'措辞不匹配的拒答' -- GT 也是拒答但用词不同。")
        print(f"  Step4 claim 若为 negation_refusal: 预期 SUPPORTED (图中确实看不出名字)。")
        print(f"  → decision=PRESERVE → 不编辑 → base_f1 稳定，比旧方法更保守。")


if __name__ == "__main__":
    main()
