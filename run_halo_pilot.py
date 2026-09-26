# -*- coding: utf-8 -*-
"""
run_halo_pilot.py — haloquest rho/span 重生成试点驱动 (修正路径版)

与 run_haloquest_pilot.py 逻辑相同，但修正了以下路径问题：
  1. DATA_PATH / DATA_ROOT 指向 REPO/OmniConfess/OmniHalluBench/... (原脚本少了一级)
  2. sys.path 确保 OmniConfess/ 子目录可找到 methods/shared/

不修改任何既有文件。

用法:
  export MMHALU_MODEL_ROOT=/slow_share/tsq/MODELS HF_HUB_OFFLINE=1
  CUDA_VISIBLE_DEVICES=0 /slow_share/tsq/venvs/omniconfess/bin/python \
      run_halo_pilot.py --limit 30 --out outputs/freeform_haloquest_pilot30.jsonl
  # 全量:
  CUDA_VISIBLE_DEVICES=0 /slow_share/tsq/venvs/omniconfess/bin/python \
      run_halo_pilot.py --limit 0 --out outputs/freeform_haloquest_full295.jsonl
"""
import argparse
import json
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.abspath(__file__))

# ---- 修正 sys.path ----
# freeform_repair 在 REPO 根
sys.path.insert(0, REPO)
# methods/shared/ 在 REPO/OmniConfess/methods/shared/
OMNICONFESS_PKG = os.path.join(REPO, "OmniConfess")
if OMNICONFESS_PKG not in sys.path:
    sys.path.insert(0, OMNICONFESS_PKG)

# ---- 修正数据路径 ----
# 实际文件在 REPO/OmniConfess/OmniHalluBench/haloquest/haloquest.json
DATA_PATH = os.path.join(REPO, "OmniConfess", "OmniHalluBench", "haloquest", "haloquest.json")
DATA_ROOT = os.path.join(REPO, "OmniConfess", "OmniHalluBench", "haloquest")

import freeform_repair as fr  # noqa: E402
from scripts.tokenf1_fullset import token_f1  # noqa: E402


def build_join():
    """cache 样本(带 spans) join haloquest.json 归一化 item(带 image_path)。key=id。"""
    from methods.shared.data_loader import load_dataset
    items = load_dataset(DATA_PATH, DATA_ROOT)
    by_id = {str(it["id"]): it for it in items}
    cache = fr.load_haloquest()
    joined = []
    missing = 0
    for s in cache:
        it = by_id.get(str(s["id"]))
        if it is None:
            missing += 1
            continue
        joined.append((s, it))
    return joined, len(cache), len(items), missing


def run(limit, out_path, tau, prior_baseline, max_new_tokens):
    cfg = fr.RepairConfig(
        tau=tau,
        prior_baseline=prior_baseline,
        max_new_tokens=max_new_tokens,
        dataset_name="haloquest",
    )
    joined, n_cache, n_data, n_missing = build_join()
    print(
        f"[join] cache={n_cache} data={n_data} joined={len(joined)} missing_in_data={n_missing}",
        flush=True,
    )
    if limit and limit > 0:
        joined = joined[:limit]

    print("[model] loading Qwen2.5-Omni-7B ...", flush=True)
    from methods.shared.model_utils import load_model
    t0 = time.time()
    processor, model = load_model()
    print(f"[model] loaded in {time.time()-t0:.1f}s", flush=True)

    rows = []
    n_fail = 0
    base_f1s, rep_f1s = [], []

    for k, (s, it) in enumerate(joined):
        cfg_s = fr.RepairConfig(
            tau=tau,
            prior_baseline=prior_baseline,
            max_new_tokens=max_new_tokens,
            dataset_name="haloquest",
        )
        scored = fr.score_sample(s, cfg_s)
        base_pred = fr.tokens_to_text(s["tokens"])
        gt = it.get("gt", "")
        spans = scored["spans"]
        rec = {
            "id": s["id"],
            "question": it.get("question", ""),
            "gt": gt,
            "n_tokens": scored["n_tokens"],
            "n_spans": len(spans),
            "spans": spans,
            "base_pred": base_pred,
            "status": "success",
        }

        if not spans:
            # 无待修订 span -> repaired == base
            rec["repaired_pred"] = base_pred
            rec["replacements"] = []
            rec["no_edit"] = True
        else:
            try:
                res = fr.regenerate_spans(
                    it, s["tokens"], spans, processor, model,
                    cfg_s, DATA_PATH, DATA_ROOT,
                )
                rec["repaired_pred"] = res["repaired_text"]
                rec["replacements"] = res["replacements"]
            except Exception as e:
                n_fail += 1
                rec["status"] = "regen_failed"
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["repaired_pred"] = base_pred  # 失败回退 base
                rec["traceback"] = traceback.format_exc()[-800:]
                print(f"[FAIL id={s['id']}] {rec['error']}", flush=True)

        rec["base_f1"] = round(100 * token_f1(base_pred, gt), 2)
        rec["repaired_f1"] = round(100 * token_f1(rec["repaired_pred"], gt), 2)
        if rec["status"] == "success":
            base_f1s.append(rec["base_f1"])
            rep_f1s.append(rec["repaired_f1"])
        rows.append(rec)

        if (k + 1) % 5 == 0 or (k + 1) == len(joined):
            print(f"[prog] {k+1}/{len(joined)} fails={n_fail}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ok = len(base_f1s)
    mb = sum(base_f1s) / n_ok if n_ok else 0.0
    mr = sum(rep_f1s) / n_ok if n_ok else 0.0
    n_edited = sum(
        1 for r in rows if r.get("n_spans", 0) > 0 and r["status"] == "success"
    )

    # 3 个 before/after 实例 (取有 span 的前 3 个)
    examples = [r for r in rows if r.get("n_spans", 0) > 0 and r["status"] == "success"][:3]
    print("\n===== 3 BEFORE/AFTER EXAMPLES =====", flush=True)
    for ex in examples:
        print(json.dumps({
            "id": ex["id"],
            "question": ex["question"],
            "base_pred": ex["base_pred"][:300],
            "repaired_pred": ex["repaired_pred"][:300],
            "base_f1": ex["base_f1"],
            "repaired_f1": ex["repaired_f1"],
            "n_spans": ex["n_spans"],
        }, ensure_ascii=False, indent=2), flush=True)

    summary = {
        "n_run": len(rows),
        "n_success": n_ok,
        "n_regen_failed": n_fail,
        "n_samples_edited": n_edited,
        "n_samples_no_span": sum(1 for r in rows if r.get("no_edit")),
        "base_tokenf1_mean": round(mb, 2),
        "repaired_tokenf1_mean": round(mr, 2),
        "delta_f1": round(mr - mb, 2),
        "tau": tau,
        "prior_baseline": prior_baseline,
        "out": out_path,
    }
    print("\n===== PILOT SUMMARY (haloquest, Token-F1) =====", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description="haloquest rho/span 重生成试点 (路径修正版)")
    ap.add_argument("--limit", type=int, default=30, help="样本数, 0=全量 295")
    ap.add_argument(
        "--out", type=str,
        default=os.path.join(REPO, "outputs", "freeform_haloquest_pilot30.jsonl"),
    )
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--prior-baseline", type=str, default="none")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()

    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    run(args.limit, out, args.tau, args.prior_baseline, args.max_new_tokens)


if __name__ == "__main__":
    main()
