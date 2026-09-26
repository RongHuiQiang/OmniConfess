#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_localizer_v2_a4.py — A4: 离线验证 claim 解析质量 + is_refusal 修复

验证点:
  1. _extract_claim 区分 object_existence vs object_attribute
  2. id=601 is_refusal 修复后是否正确激活
  3. 前20个有候选的样本 claim_type 分布

运行:
    cd /home/tsq/rhq/omniconfess/OmniConfess
    python3 scripts/validate_localizer_v2_a4.py 2>&1 | tee outputs/localizer_v2_diag/validate_a4.txt
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import localizer_v2 as lv2

CACHE_PATH = os.path.join(REPO, "outputs", "cf_probe", "haloquest_textcf_tokens.jsonl")
OUT_DIR    = os.path.join(REPO, "outputs", "localizer_v2_diag")
os.makedirs(OUT_DIR, exist_ok=True)


# ── helper ───────────────────────────────────────────────────────────────────

def load_first_n(n: int) -> list[dict]:
    records = []
    with open(CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            if len(records) >= n:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_by_ids(ids: set) -> dict[str, dict]:
    out = {}
    with open(CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = str(r.get("id", ""))
            if rid in ids:
                out[rid] = r
    return out


# ── Test 1: unit tests for _extract_claim ───────────────────────────────────

def test_extract_claim():
    print("=" * 72)
    print("TEST 1: _extract_claim unit tests")
    print("=" * 72)

    cases = [
        # (span_text, question, base_pred, expected_claim_type)
        ("the red beanie",       "What color is the hat?",  "",  "object_attribute"),
        ("the beanie",           "What is on the shelf?",   "",  "object_existence"),
        ("two dogs",             "How many dogs are there?","",  "object_count"),
        ("three red balloons",   "What is in the image?",   "",  "object_count"),   # count wins over attr
        ("not visible",          "What is in the image?",   "",  "answerability"),
        ("cannot determine",     "Who is the person?",      "",  "answerability"),
        ("standing on the left", "Where is the person?",    "",  "action"),
        ("a large wooden table", "What furniture is shown?","",  "object_attribute"),
        ("the bicycle",          "What vehicle is shown?",  "",  "object_existence"),
        ("blue shirt",           "What is the man wearing?","",  "object_attribute"),
    ]

    pass_count = 0
    for span_text, question, base_pred, expected in cases:
        result = lv2._extract_claim(span_text, question, base_pred)
        ok = result["claim_type"] == expected
        status = "PASS" if ok else "FAIL"
        if ok:
            pass_count += 1
        print(f"  [{status}] span={span_text!r:<30s}  "
              f"expected={expected:<20s}  got={result['claim_type']}")
        if not ok or result.get("attribute_claim"):
            print(f"         primary  : {result['primary_claim']}")
            if result.get("attribute_claim"):
                print(f"         attribute: {result['attribute_claim']}")

    print(f"\n  Result: {pass_count}/{len(cases)} passed")
    return pass_count, len(cases)


# ── Test 2: is_refusal fix for id=601 ────────────────────────────────────────

def test_is_refusal_fix():
    print()
    print("=" * 72)
    print("TEST 2: is_refusal fix — id=601 and similar refusal samples")
    print("=" * 72)

    # id=601 is the known broken case: ' not' was separated from its span
    # by gap-merge logic
    target_ids = {"601", "179", "608"}
    recs = load_by_ids(target_ids)

    results = {}
    for sid in sorted(target_ids):
        rec = recs.get(sid)
        if rec is None:
            print(f"  id={sid}: NOT FOUND in cache")
            continue

        base_pred = "".join(rec.get("tokens", []))
        lr = lv2.localize(rec)

        any_refusal = any(c.is_refusal for c in lr.candidates)
        refusal_cands = [c for c in lr.candidates if c.is_refusal]

        # is the base_pred itself a refusal?
        _REFUSAL_PHRASES = (
            "not ", "no ", "cannot", "can't", "isn't", "aren't", "wasn't",
            "never", "without", "unclear", "unknown", "unable", "nothing",
            "none", "lack", "don't know",
        )
        pred_is_refusal = any(ph in base_pred.lower() for ph in _REFUSAL_PHRASES)

        print(f"\n  id={sid}")
        print(f"    base_pred     : {base_pred[:100]}")
        print(f"    pred_is_refusal (heuristic): {pred_is_refusal}")
        print(f"    n_candidates  : {len(lr.candidates)}")
        print(f"    is_refusal activated: {any_refusal}")
        for c in lr.candidates:
            marker = " <-- REFUSAL" if c.is_refusal else ""
            print(f"      span={c.span_text!r:<40s}  kind={c.span_kind:<20s}{marker}")
            if c.is_refusal:
                print(f"        claim: {c.claim[:80]}")

        # For id=601 specifically: check the preceding_token mechanism
        if sid == "601":
            tokens = rec.get("tokens", [])
            print(f"\n    [id=601 token-level inspection]")
            for i, (tok, dv) in enumerate(zip(tokens, rec.get("delta_img_tok", []))):
                print(f"      [{i:2d}] {tok!r:<20s}  delta_img={dv:+.4f}")
            print(f"\n    Expected: is_refusal=True on a span near ' not' token")
            status = "PASS" if any_refusal else "FAIL (is_refusal still not activated)"
            print(f"    Status: {status}")

        results[sid] = {
            "pred_is_refusal": pred_is_refusal,
            "is_refusal_activated": any_refusal,
            "n_refusal_cands": len(refusal_cands),
        }

    return results


# ── Test 3: claim_type distribution over first 20 samples with candidates ───

def test_claim_distribution():
    print()
    print("=" * 72)
    print("TEST 3: claim_type distribution — first 20 samples with candidates")
    print("=" * 72)

    records = load_first_n(200)  # load enough to get 20 with candidates
    claim_type_counter: Counter = Counter()
    span_kind_counter:  Counter = Counter()
    samples_with_cands = 0
    shown = 0

    per_sample_rows = []
    for rec in records:
        if samples_with_cands >= 20:
            break
        lr = lv2.localize(rec)
        if not lr.candidates:
            continue
        samples_with_cands += 1
        sid = str(rec.get("id", "?"))
        base_pred = lr.base_pred[:60]

        for c in lr.candidates:
            ct = c.claim_parsed.get("claim_type", "missing")
            claim_type_counter[ct] += 1
            span_kind_counter[c.span_kind] += 1

        row = {
            "id": sid,
            "base_pred_snippet": base_pred,
            "n_cands": len(lr.candidates),
            "claim_types": [c.claim_parsed.get("claim_type","?") for c in lr.candidates],
            "span_kinds":  [c.span_kind for c in lr.candidates],
        }
        per_sample_rows.append(row)

        if shown < 20:
            print(f"\n  id={sid}  base_pred={base_pred!r}")
            for c in lr.candidates:
                ct = c.claim_parsed.get("claim_type", "?")
                pri = c.claim_parsed.get("primary_claim", "")[:60]
                att = c.claim_parsed.get("attribute_claim", "")[:50]
                print(f"    span={c.span_text!r:<30s}  claim_type={ct:<20s}  "
                      f"vis_dep={c.visual_dependence:.3f}")
                print(f"      primary  : {pri}")
                if att:
                    print(f"      attribute: {att}")
            shown += 1

    print()
    print(f"  Samples with candidates found: {samples_with_cands}")
    print()
    print("  claim_type distribution:")
    total_claims = sum(claim_type_counter.values())
    for ct, cnt in claim_type_counter.most_common():
        pct = 100.0 * cnt / max(total_claims, 1)
        print(f"    {ct:<25s} {cnt:4d}  ({pct:.1f}%)")

    print()
    print("  span_kind distribution (cross-check):")
    for sk, cnt in span_kind_counter.most_common():
        pct = 100.0 * cnt / max(total_claims, 1)
        print(f"    {sk:<25s} {cnt:4d}  ({pct:.1f}%)")

    # Write detailed results
    out_path = os.path.join(OUT_DIR, "validate_claim_dist.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in per_sample_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n  Detailed distribution written to {out_path}")

    return claim_type_counter, span_kind_counter


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("[A4] validate_localizer_v2_a4.py — offline validation\n")

    pass_ct, total_ct = test_extract_claim()
    refusal_results    = test_is_refusal_fix()
    claim_dist, _      = test_claim_distribution()

    # ── summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("A4 VALIDATION SUMMARY")
    print("=" * 72)

    print(f"\n  _extract_claim unit tests : {pass_ct}/{total_ct} passed")

    print("\n  is_refusal fix results:")
    for sid, r in sorted(refusal_results.items()):
        activated = r.get("is_refusal_activated", False)
        pred_ref  = r.get("pred_is_refusal", False)
        status = "OK" if (not pred_ref or activated) else "NEEDS ATTENTION"
        print(f"    id={sid}  pred_refusal={pred_ref}  "
              f"is_refusal_activated={activated}  [{status}]")

    print("\n  claim_type sanity:")
    total = sum(claim_dist.values())
    for ct, cnt in claim_dist.most_common():
        print(f"    {ct:<25s} {cnt:3d} / {total}")

    # Key assertions
    print()
    has_attr = claim_dist.get("object_attribute", 0) > 0
    has_exist = claim_dist.get("object_existence", 0) > 0
    separation_ok = has_attr and has_exist
    print(f"  [{'PASS' if separation_ok else 'FAIL'}] "
          f"object_attribute and object_existence both appear (separation check)")
    print(f"  [{'PASS' if pass_ct >= 7 else 'FAIL'}] "
          f"_extract_claim >= 7/10 unit tests passed")

    print()
    print("[A4] Done.")


if __name__ == "__main__":
    main()
