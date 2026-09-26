"""
repair_harness.py — HaloQuest repair 实验对比 Harness

功能：
1. 对比两个 .jsonl 结果文件，输出 paired 指标
2. 自动检测 Champion（任何结果优于当前最优就标记）
3. 保存每次运行的 metadata（配置、时间、关键指标）
4. 分类失败案例

用法：
  python3 scripts/repair_harness.py \\
    --baseline outputs/lv2_ablation/arm_lv2_full.jsonl \\
    --challenger outputs/lv2_v2/arm_v2_n50.jsonl \\
    --save-report outputs/harness/report_$(date +%Y%m%d_%H%M).json
"""
import os
import sys
import json
import argparse
import datetime
from collections import defaultdict


# ---------------------------------------------------------------------------
# Token F1 — multiref variant matching freeform_repair.py
# ---------------------------------------------------------------------------

def _normalize(s: str) -> list:
    """Lowercase, strip punctuation, split on whitespace."""
    import re
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return s.split()


def _single_f1(pred_tokens: list, ref_tokens: list) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    common = pred_set & ref_set
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(ref_tokens)
    return 2 * prec * rec / (prec + rec)


def token_f1_multiref(pred: str, gt_str: str) -> float:
    """
    Compute token-F1 between pred and the best-matching reference in gt_str.
    gt_str is ';'-separated multi-reference string (same convention as
    freeform_repair.py / tokenf1_fullset.py).
    Returns score in [0, 100].
    """
    pred_tokens = _normalize(pred)
    refs = [r.strip() for r in gt_str.split(";") if r.strip()]
    if not refs:
        return 0.0
    best = max(_single_f1(pred_tokens, _normalize(r)) for r in refs)
    return round(best * 100, 2)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_results(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------

_REFUSAL_WORDS = {"not visible", "cannot", "can't", "no ", "there is no",
                  "not present", "not shown", "unable", "unknown", "unclear",
                  "don't know", "do not know", "cannot see"}


def _is_refusal(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in _REFUSAL_WORDS)


def _classify_failure(base_f1: float, repaired_f1: float,
                      base_pred: str) -> str:
    """
    Classify a degradation sample (repaired_f1 < base_f1) into a named
    failure type.
    """
    if _is_refusal(base_pred) and base_f1 > 50:
        return "correct_refusal_degraded"
    if base_f1 > 70:
        return "unnecessary_edit"
    return "other_degradation"


def _classify_improvement(base_f1: float, repaired_f1: float,
                           base_pred: str) -> str:
    """Name notable improvement types (informational only)."""
    if base_f1 < 30 and repaired_f1 > base_f1 + 10:
        return "hallucination_corrected"
    return "improvement"


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compare(baseline_rows: list, challenger_rows: list,
            label_base: str = "baseline",
            label_new: str = "challenger") -> dict:
    """
    Align by sample id and compute paired metrics.

    Returns a dict with:
      - per_sample: list of per-sample dicts
      - summary: high-level stats
      - failure_cases: categorised degradations
      - improvement_cases: categorised gains
    """
    # Build lookup by id (str)
    base_by_id = {str(r.get("id", r.get("qid", i))): r
                  for i, r in enumerate(baseline_rows)}
    chal_by_id = {str(r.get("id", r.get("qid", i))): r
                  for i, r in enumerate(challenger_rows)}

    common_ids = sorted(set(base_by_id) & set(chal_by_id), key=lambda x: int(x) if x.isdigit() else x)
    only_base = set(base_by_id) - set(chal_by_id)
    only_chal = set(chal_by_id) - set(base_by_id)

    per_sample = []
    delta_list = []
    oracle_delta_list = []
    failure_cases = defaultdict(list)
    improvement_cases = defaultdict(list)

    for sid in common_ids:
        b = base_by_id[sid]
        c = chal_by_id[sid]

        # Pull F1 from pre-computed field if present, else recompute
        b_f1 = b.get("base_f1", None)
        b_rep_f1 = b.get("repaired_f1", None)
        c_f1 = c.get("base_f1", None)
        c_rep_f1 = c.get("repaired_f1", None)

        # Determine "effective" F1 for each file (what was actually chosen)
        # Baseline file: use its accepted prediction
        if b_f1 is not None and b_rep_f1 is not None:
            b_accepted = b.get("lv2_regen_accepted", True)
            b_effective_f1 = b_rep_f1 if b_accepted else b_f1
            b_oracle_f1 = max(b_f1, b_rep_f1)
        else:
            # Fallback: compute from pred field
            gt = b.get("gt", "")
            base_pred = b.get("base_pred", b.get("pred", ""))
            repaired_pred = b.get("repaired_pred", base_pred)
            b_f1 = token_f1_multiref(base_pred, gt)
            b_rep_f1 = token_f1_multiref(repaired_pred, gt)
            b_accepted = b.get("lv2_regen_accepted", True)
            b_effective_f1 = b_rep_f1 if b_accepted else b_f1
            b_oracle_f1 = max(b_f1, b_rep_f1)

        # Challenger file
        if c_f1 is not None and c_rep_f1 is not None:
            c_accepted = c.get("lv2_regen_accepted", True)
            c_effective_f1 = c_rep_f1 if c_accepted else c_f1
            c_oracle_f1 = max(c_f1, c_rep_f1)
        else:
            gt = c.get("gt", "")
            base_pred = c.get("base_pred", c.get("pred", ""))
            repaired_pred = c.get("repaired_pred", base_pred)
            c_f1 = token_f1_multiref(base_pred, gt)
            c_rep_f1 = token_f1_multiref(repaired_pred, gt)
            c_accepted = c.get("lv2_regen_accepted", True)
            c_effective_f1 = c_rep_f1 if c_accepted else c_f1
            c_oracle_f1 = max(c_f1, c_rep_f1)

        delta = c_effective_f1 - b_effective_f1
        oracle_delta = c_oracle_f1 - b_oracle_f1
        delta_list.append(delta)
        oracle_delta_list.append(oracle_delta)

        # Classify
        base_pred_text = b.get("base_pred", b.get("pred", ""))
        if delta < 0:
            ftype = _classify_failure(b_f1, c_effective_f1, base_pred_text)
            failure_cases[ftype].append({
                "id": sid,
                "base_f1": b_effective_f1,
                "challenger_f1": c_effective_f1,
                "delta": round(delta, 2),
                "base_pred": base_pred_text[:120],
            })
        elif delta > 0:
            itype = _classify_improvement(b_f1, c_effective_f1, base_pred_text)
            improvement_cases[itype].append({
                "id": sid,
                "base_f1": b_effective_f1,
                "challenger_f1": c_effective_f1,
                "delta": round(delta, 2),
            })

        per_sample.append({
            "id": sid,
            f"{label_base}_f1": round(b_effective_f1, 2),
            f"{label_new}_f1": round(c_effective_f1, 2),
            "delta": round(delta, 2),
            "oracle_delta": round(oracle_delta, 2),
        })

    n = len(common_ids)
    mean_delta = sum(delta_list) / n if n else 0.0
    mean_oracle_delta = sum(oracle_delta_list) / n if n else 0.0

    improve_count = sum(1 for d in delta_list if d > 0)
    regress_count = sum(1 for d in delta_list if d < 0)
    tie_count = sum(1 for d in delta_list if d == 0)

    summary = {
        "n_common": n,
        "n_only_baseline": len(only_base),
        "n_only_challenger": len(only_chal),
        "mean_delta_f1": round(mean_delta, 4),
        "mean_oracle_delta_f1": round(mean_oracle_delta, 4),
        "improve": improve_count,
        "regress": regress_count,
        "tie": tie_count,
        "failure_breakdown": {k: len(v) for k, v in failure_cases.items()},
        "improvement_breakdown": {k: len(v) for k, v in improvement_cases.items()},
    }

    return {
        "summary": summary,
        "per_sample": per_sample,
        "failure_cases": dict(failure_cases),
        "improvement_cases": dict(improvement_cases),
    }


# ---------------------------------------------------------------------------
# Champion detection
# ---------------------------------------------------------------------------

CHAMPION_FILE = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "harness", "champion.json"
)


def load_champion() -> dict:
    path = os.path.abspath(CHAMPION_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"mean_delta_f1": -999.0, "run_label": "none", "run_time": ""}


def maybe_update_champion(summary: dict, run_label: str, report_path: str) -> bool:
    """Return True if challenger is a new champion."""
    champ = load_champion()
    if summary["mean_delta_f1"] > champ["mean_delta_f1"]:
        new_champ = {
            "mean_delta_f1": summary["mean_delta_f1"],
            "mean_oracle_delta_f1": summary["mean_oracle_delta_f1"],
            "run_label": run_label,
            "run_time": datetime.datetime.now().isoformat(),
            "report_path": report_path,
        }
        os.makedirs(os.path.dirname(os.path.abspath(CHAMPION_FILE)), exist_ok=True)
        with open(os.path.abspath(CHAMPION_FILE), "w") as f:
            json.dump(new_champ, f, indent=2)
        return True
    return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compare two HaloQuest repair .jsonl result files."
    )
    ap.add_argument("--baseline", required=True,
                    help="Path to baseline results .jsonl")
    ap.add_argument("--challenger", required=True,
                    help="Path to challenger results .jsonl")
    ap.add_argument("--label-base", default="baseline",
                    help="Human label for baseline run")
    ap.add_argument("--label-new", default="challenger",
                    help="Human label for challenger run")
    ap.add_argument("--save-report", default=None,
                    help="If set, save JSON report to this path")
    ap.add_argument("--champion", action="store_true",
                    help="Check/update champion tracker")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-sample table")
    args = ap.parse_args()

    baseline_rows = load_results(args.baseline)
    challenger_rows = load_results(args.challenger)

    result = compare(baseline_rows, challenger_rows,
                     label_base=args.label_base,
                     label_new=args.label_new)

    s = result["summary"]
    print(f"\n=== Harness comparison: {args.label_base} vs {args.label_new} ===")
    print(f"  Samples aligned : {s['n_common']}")
    print(f"  Mean ΔF1        : {s['mean_delta_f1']:+.4f}")
    print(f"  Oracle ΔF1      : {s['mean_oracle_delta_f1']:+.4f}")
    print(f"  Improve / Tie / Regress : {s['improve']} / {s['tie']} / {s['regress']}")
    if s["failure_breakdown"]:
        print("  Failure breakdown:")
        for k, v in s["failure_breakdown"].items():
            print(f"    {k}: {v}")

    if args.verbose:
        print(f"\n{'id':>6}  {args.label_base:>12}  {args.label_new:>12}  {'delta':>8}")
        for row in result["per_sample"]:
            print(f"{row['id']:>6}  "
                  f"{row[args.label_base + '_f1']:>12.2f}  "
                  f"{row[args.label_new + '_f1']:>12.2f}  "
                  f"{row['delta']:>+8.2f}")

    report_path = args.save_report
    if report_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        report = {
            "meta": {
                "baseline": args.baseline,
                "challenger": args.challenger,
                "label_base": args.label_base,
                "label_new": args.label_new,
                "run_time": datetime.datetime.now().isoformat(),
            },
            **result,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  Report saved to: {report_path}")

    if args.champion and report_path:
        is_new = maybe_update_champion(s, args.label_new, report_path)
        if is_new:
            print(f"  *** NEW CHAMPION: {args.label_new} ({s['mean_delta_f1']:+.4f}) ***")
        else:
            champ = load_champion()
            print(f"  Champion still: {champ['run_label']} ({champ['mean_delta_f1']:+.4f})")


if __name__ == "__main__":
    main()
