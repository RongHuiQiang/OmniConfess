#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_localizer_v2_full.py — A1: 对前50个样本逐样本诊断 localizer_v2

分析项目:
  - 选中的 span 文本 / span_kind / visual_dependence
  - is_refusal 激活情况
  - token 级 visual_dep 分布 (max/min/mean)
  - 分类: 精准定位 / 误定位 / 漏定位

运行:
    cd /home/tsq/rhq/omniconfess/OmniConfess
    python3 scripts/diagnose_localizer_v2_full.py 2>&1 | tee outputs/localizer_v2_diag/diag_full.txt
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import localizer_v2 as lv2

CACHE_PATH = os.path.join(REPO, "outputs", "cf_probe", "haloquest_textcf_tokens.jsonl")
OUT_DIR    = os.path.join(REPO, "outputs", "localizer_v2_diag")
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLES = 50


# ── negation heuristic (ground-truth proxy, offline) ─────────────────────────
_REFUSAL_PHRASES = (
    "not ", "no ", "cannot", "can't", "isn't", "aren't", "wasn't",
    "weren't", "never", "without", "absent", "missing", "unclear",
    "unknown", "unable", "nothing", "none", "lack", "don't know",
    "not visible", "not clear", "not possible", "not sure",
    "not available", "not enough",
)

def _pred_has_refusal(base_pred: str) -> bool:
    bp = base_pred.lower()
    return any(ph in bp for ph in _REFUSAL_PHRASES)


def _span_covers_error(span_text: str, neg_item) -> bool:
    """
    Proxy: does span overlap with the neg_item (the confabulated claim)?
    neg_item is the wrong entity/attribute from the probe record.
    """
    if not neg_item or not isinstance(neg_item, str):
        return False
    s = span_text.lower().strip()
    n = neg_item.lower().strip()
    # check overlap of significant words
    s_words = set(w.strip(".,!?;:'\"") for w in s.split() if len(w) > 2)
    n_words = set(w.strip(".,!?;:'\"") for w in n.split() if len(w) > 2)
    return bool(s_words & n_words)


def _classify_sample(record: dict, candidates) -> str:
    """
    Classify this sample's localization outcome:
      precise   — at least one span overlaps with neg_item
      mislocate — candidates exist but none overlap neg_item
      miss      — no candidates (delta_img_tok too low or coverage dropped)
    """
    neg_item = record.get("neg_item", "")
    if not candidates:
        return "miss"
    for c in candidates:
        if _span_covers_error(c.span_text, neg_item):
            return "precise"
    return "mislocate"


def main():
    print(f"[A1] Loading {CACHE_PATH} ...", flush=True)
    records = []
    with open(CACHE_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= N_SAMPLES:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[A1] Loaded {len(records)} records.\n", flush=True)

    # ── per-sample stats ─────────────────────────────────────────────────────
    stats = {
        "precise": [],
        "mislocate": [],
        "miss": [],
        "refusal_activated": [],
        "refusal_missed": [],        # pred has refusal but is_refusal not set
    }
    span_kind_counter: Counter = Counter()
    vis_dep_vals: list[float] = []
    results_jsonl: list[dict] = []

    for rec in records:
        rid  = str(rec.get("id", "?"))
        qtxt = rec.get("question", "")
        base_pred = "".join(rec.get("tokens", []))
        neg_item  = rec.get("neg_item", "")
        delta_img = np.asarray(rec.get("delta_img_tok", []), dtype=float)

        lr = lv2.localize(rec)
        cands = lr.candidates

        # token-level distribution
        tok_max  = float(delta_img.max())  if delta_img.size else 0.0
        tok_min  = float(delta_img.min())  if delta_img.size else 0.0
        tok_mean = float(delta_img.mean()) if delta_img.size else 0.0
        vis_dep_vals.extend(delta_img.tolist())

        outcome = _classify_sample(rec, cands)
        stats[outcome].append(rid)

        # refusal diagnosis
        pred_refusal = _pred_has_refusal(base_pred)
        cand_refusal = any(c.is_refusal for c in cands)
        if pred_refusal and cand_refusal:
            stats["refusal_activated"].append(rid)
        elif pred_refusal and not cand_refusal:
            stats["refusal_missed"].append(rid)

        for c in cands:
            span_kind_counter[c.span_kind] += 1

        # build output record
        results_jsonl.append({
            "id": rid,
            "question": qtxt,
            "base_pred": base_pred,
            "neg_item": neg_item,
            "outcome": outcome,
            "tok_vis_max": round(tok_max, 4),
            "tok_vis_min": round(tok_min, 4),
            "tok_vis_mean": round(tok_mean, 4),
            "n_candidates": len(cands),
            "pred_has_refusal": pred_refusal,
            "any_is_refusal": cand_refusal,
            "candidates": [
                {
                    "span_text": c.span_text,
                    "span_kind": c.span_kind,
                    "visual_dependence": round(c.visual_dependence, 4),
                    "is_refusal": c.is_refusal,
                    "tok_vis_dep": [round(v, 3) for v in c.tok_visual_dependence],
                    "claim": c.claim,
                    "claim_parsed": c.claim_parsed,
                }
                for c in cands
            ],
        })

    # ── write jsonl ──────────────────────────────────────────────────────────
    out_jsonl = os.path.join(OUT_DIR, "diag_samples.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results_jsonl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── print summary ────────────────────────────────────────────────────────
    total = len(records)
    print("=" * 72)
    print("A1 DIAGNOSIS SUMMARY  (first 50 samples)")
    print("=" * 72)
    print(f"  Total samples analysed  : {total}")
    print(f"  Precise localisation    : {len(stats['precise']):3d}  "
          f"(span overlaps neg_item)")
    print(f"  Mis-localised           : {len(stats['mislocate']):3d}  "
          f"(candidates exist but miss neg_item)")
    print(f"  No candidates (miss)    : {len(stats['miss']):3d}  "
          f"(delta_img_tok all low)")
    print()
    print(f"  Refusal pred+activated  : {len(stats['refusal_activated']):3d}")
    print(f"  Refusal pred but MISSED : {len(stats['refusal_missed']):3d}  "
          f"(is_refusal not set)")
    print()
    print("  span_kind distribution (across all candidates):")
    for kind, cnt in span_kind_counter.most_common():
        print(f"    {kind:<25s} {cnt:4d}")
    print()
    arr = np.asarray(vis_dep_vals)
    print(f"  Global delta_img_tok stats (all tokens, {len(arr)} total):")
    print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}")
    pct_pos = (arr > 0).mean() * 100
    pct_hi  = (arr > 0.05).mean() * 100
    print(f"    % positive        : {pct_pos:.1f}%")
    print(f"    % > 0.05 (high)   : {pct_hi:.1f}%")

    # ── print representative examples ────────────────────────────────────────
    def _show_examples(label: str, ids: list[str], n: int = 3):
        print()
        print(f"--- {label} (up to {n} examples) ---")
        shown = 0
        for r in results_jsonl:
            if r["id"] in ids and shown < n:
                print(f"  id={r['id']}  neg_item={r['neg_item']!r}")
                print(f"  base_pred : {r['base_pred'][:120]}")
                for c in r["candidates"][:3]:
                    print(f"    span={c['span_text']!r:35s}  "
                          f"kind={c['span_kind']:<22s}  "
                          f"vd={c['visual_dependence']:.3f}  "
                          f"refusal={c['is_refusal']}")
                if not r["candidates"]:
                    print("    (no candidates)")
                shown += 1

    _show_examples("PRECISE", stats["precise"])
    _show_examples("MIS-LOCALISED", stats["mislocate"])
    _show_examples("MISSED (no candidates)", stats["miss"])
    if stats["refusal_missed"]:
        _show_examples("REFUSAL PRED BUT is_refusal NOT SET", stats["refusal_missed"])

    print()
    print(f"[A1] Detailed results written to {out_jsonl}")
    print("[A1] Done.")


if __name__ == "__main__":
    main()
