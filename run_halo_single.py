# -*- coding: utf-8 -*-
"""
run_halo_single.py — haloquest 单 tau 单 GPU 驱动 (v3, 新文件)

修复关键顺序问题:
  1. build_join() 在 load_model() 之前调用, 确保路径/数据错误快速报出
  2. 路径使用 REPO/OmniHalluBench/... (无多余 OmniConfess 层)
  3. 第一行就 print 进程信息, 确保 log 有输出
  4. 兼容 tau=inf (不进行 Rev_theta, 直接用 base_pred 打分)

约束: 只能用卡 0-3 (卡 4-7 他人进程禁碰)。不修改任何既有文件。

用法:
  CUDA_VISIBLE_DEVICES=0 MMHALU_MODEL_ROOT=/slow_share/tsq/MODELS HF_HUB_OFFLINE=1 \
    /slow_share/tsq/venvs/omniconfess/bin/python run_halo_single.py \
    --tau 0.5 --limit 50 --out outputs/freeform_grid/haloquest_tau0.5.jsonl
"""
import argparse
import json
import math
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "OmniConfess"))

# 正确数据路径 (REPO/OmniHalluBench, 无多余 OmniConfess 层)
DATA_PATH = os.path.join(REPO, "OmniHalluBench", "haloquest", "haloquest.json")
DATA_ROOT = os.path.join(REPO, "OmniHalluBench", "haloquest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--prior-baseline", type=str, default="none")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()

    tau = args.tau
    is_inf = math.isinf(tau) or tau > 1e8

    out_path = args.out
    if not out_path:
        tag = "inf" if is_inf else str(tau)
        out_path = os.path.join(REPO, "outputs", "freeform_grid",
                                f"haloquest_tau{tag}.jsonl")
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO, out_path)

    # --- 第一件事: 立即打印进程信息到 log ---
    print(f"[start] pid={os.getpid()} tau={tau} is_inf={is_inf} "
          f"limit={args.limit} out={out_path}", flush=True)
    print(f"[path] DATA_PATH={DATA_PATH}", flush=True)
    print(f"[path] DATA exists: {os.path.exists(DATA_PATH)}", flush=True)

    # --- 导入 (NFS 慢导入在这里) ---
    print("[import] loading freeform_repair...", flush=True)
    import freeform_repair as fr
    print("[import] loading tokenf1...", flush=True)
    from scripts.tokenf1_fullset import token_f1
    print("[import] done", flush=True)

    # --- build_join 在 load_model 之前 ---
    print("[join] building join...", flush=True)
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
    n_cache, n_data = len(cache), len(items)
    print(f"[join] cache={n_cache} data={n_data} joined={len(joined)} missing={missing}",
          flush=True)

    if args.limit and args.limit > 0:
        joined = joined[:args.limit]
    print(f"[join] running on {len(joined)} samples (tau={tau})", flush=True)

    # --- 加载模型 (tau=inf 不需要模型) ---
    processor, model = None, None
    if not is_inf:
        print("[model] loading Qwen2.5-Omni-7B...", flush=True)
        from methods.shared.model_utils import load_model
        t0 = time.time()
        processor, model = load_model()
        print(f"[model] loaded in {time.time()-t0:.1f}s", flush=True)
    else:
        print("[model] tau=inf, skip model load", flush=True)

    # --- 主循环 ---
    rows = []
    n_fail = 0
    base_f1s, rep_f1s = [], []

    for k, (s, it) in enumerate(joined):
        cfg_s = fr.RepairConfig(
            tau=tau if not is_inf else 1e9,
            prior_baseline=args.prior_baseline,
            max_new_tokens=args.max_new_tokens,
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
            "tau": tau,
        }

        if is_inf or not spans:
            # tau=inf 或无 span: 直接用 base_pred
            rec["repaired_pred"] = base_pred
            rec["replacements"] = []
            if not spans:
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
                rec["repaired_pred"] = base_pred
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
    n_edited = sum(1 for r in rows if r.get("n_spans", 0) > 0
                   and r["status"] == "success" and not is_inf)

    # 3 before/after examples
    examples = [r for r in rows if r.get("n_spans", 0) > 0
                and r["status"] == "success"][:3]
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
        "tau": "inf" if is_inf else tau,
        "prior_baseline": args.prior_baseline,
        "out": out_path,
    }
    print("\n===== SUMMARY =====", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
