# -*- coding: utf-8 -*-
"""
run_haloquest_pilot.py — haloquest 上 rho/span 重生成 (Rev_theta) 的 GPU 闭环驱动 (新文件)

流程:
  1. freeform_repair.load_haloquest() 读缓存 per-token per-view 分数 -> score_sample 出 spans。
  2. 按 source.orig_id == cache id 关联 haloquest.json (load_dataset 归一化, 拿 image_path)。
  3. 加载 Qwen2.5-Omni-7B, 对每个样本用 regenerate_spans 只重写标记 span、未标记保留。
  4. base_pred = 缓存 tokens 还原文本; repaired_pred = 重写后文本; gt = 数据 answer。
  5. 用官方 Token-F1 (scripts/tokenf1_fullset.token_f1) 给 base 与 repaired 分别打分对比。

不打分需要金标 span 的指标 (无 GT span)。GAVIE 需 GLM API key, 本环境未提供 -> 冻结,
只报 Token-F1 (与项目现有 'GAVIE frozen' 约定一致); 输出 jsonl 可供主控事后补 GAVIE。

用法:
  export MMHALU_MODEL_ROOT=/slow_share/tsq/MODELS HF_HUB_OFFLINE=1
  CUDA_VISIBLE_DEVICES=0 /slow_share/tsq/venvs/omniconfess/bin/python \
      run_haloquest_pilot.py --limit 30 --out outputs/freeform_haloquest_pilot30.jsonl
  # 全量: --limit 0
"""
import argparse
import json
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "OmniConfess"))

import freeform_repair as fr  # noqa: E402
from scripts.tokenf1_fullset import token_f1  # noqa: E402

DATA_PATH = os.path.join(REPO, "OmniHalluBench/haloquest/haloquest.json")
DATA_ROOT = os.path.join(REPO, "OmniHalluBench/haloquest")

def build_join():
    """cache 样本(带 spans) join haloquest.json 归一化 item(带 image_path)。key=orig_id。"""
    from methods.shared.data_loader import load_dataset
    items = load_dataset(DATA_PATH, DATA_ROOT)
    # 归一化 item 的 'id' 来自 source.orig_id (见 _alias_legacy), 与 cache id 同底
    by_id = {str(it["id"]): it for it in items}
    cache = fr.load_haloquest()
    joined = []
    for s in cache:
        it = by_id.get(str(s["id"]))
        if it is None:
            continue
        joined.append((s, it))
    return joined, len(cache), len(items)


def run(limit, out_path, tau, prior_baseline, max_new_tokens):
    cfg = fr.RepairConfig(tau=tau, prior_baseline=prior_baseline,
                          max_new_tokens=max_new_tokens, dataset_name="haloquest")
    joined, n_cache, n_data = build_join()
    print(f"[join] cache={n_cache} data={n_data} joined={len(joined)}", flush=True)
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
        cfg_s = fr.RepairConfig(tau=tau, prior_baseline=prior_baseline,
                                max_new_tokens=max_new_tokens, dataset_name="haloquest")
        scored = fr.score_sample(s, cfg_s)
        base_pred = fr.tokens_to_text(s["tokens"])
        gt = it.get("gt", "")
        spans = scored["spans"]
        rec = {
            "id": s["id"], "question": it.get("question", ""), "gt": gt,
            "n_tokens": scored["n_tokens"], "n_spans": len(spans), "spans": spans,
            "base_pred": base_pred, "status": "success",
        }
        if not spans:
            # 无待修订 span -> repaired == base (照实, 不重生成)
            rec["repaired_pred"] = base_pred
            rec["replacements"] = []
            rec["no_edit"] = True
        else:
            try:
                res = fr.regenerate_spans(it, s["tokens"], spans, processor, model,
                                          cfg_s, DATA_PATH, DATA_ROOT)
                rec["repaired_pred"] = res["repaired_text"]
                rec["replacements"] = res["replacements"]
            except Exception as e:
                n_fail += 1
                rec["status"] = "regen_failed"
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["repaired_pred"] = base_pred  # 失败回退到 base, 如实标注
                rec["traceback"] = traceback.format_exc()[-800:]
                print(f"[FAIL id={s['id']}] {rec['error']}", flush=True)
        rec["base_f1"] = round(100 * token_f1(base_pred, gt), 2)
        rec["repaired_f1"] = round(100 * token_f1(rec["repaired_pred"], gt), 2)
        if rec["status"] == "success":
            base_f1s.append(rec["base_f1"])
            rep_f1s.append(rec["repaired_f1"])
        rows.append(rec)
        if (k + 1) % 10 == 0:
            print(f"[prog] {k+1}/{len(joined)} fails={n_fail}", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ok = len(base_f1s)
    mb = sum(base_f1s) / n_ok if n_ok else 0.0
    mr = sum(rep_f1s) / n_ok if n_ok else 0.0
    n_edited = sum(1 for r in rows if r.get("n_spans", 0) > 0 and r["status"] == "success")
    print("\n===== PILOT SUMMARY (haloquest, Token-F1) =====", flush=True)
    print(json.dumps({
        "n_run": len(rows), "n_success": n_ok, "n_regen_failed": n_fail,
        "n_samples_edited": n_edited,
        "base_tokenf1_mean": round(mb, 2),
        "repaired_tokenf1_mean": round(mr, 2),
        "delta_f1": round(mr - mb, 2),
        "tau": tau, "prior_baseline": prior_baseline, "out": out_path,
    }, ensure_ascii=False, indent=2), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30, help="0=全量")
    ap.add_argument("--out", type=str, default="outputs/freeform_haloquest_pilot30.jsonl")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--prior-baseline", type=str, default="none")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()
    run(args.limit, os.path.join(REPO, args.out), args.tau,
        args.prior_baseline, args.max_new_tokens)


if __name__ == "__main__":
    main()

