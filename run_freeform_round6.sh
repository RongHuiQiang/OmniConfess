#!/bin/bash
# run_freeform_round6.sh — Round 6: Phase A (BUGFIX baseline) + Phase B (evidence-constrained)
#
# 背景:
#   R1-R5 全部基于 single-ref token_f1 评估 + 两个隐含 bug:
#     Bug A1: 模型把 <<< / >>> 原样复制进 repl (marker residual)
#     Bug A2: BPE token list 里插入 decoded string 导致第二个 span 的索引偏移
#     Bug A3: token_f1 不对 HaloQuest 多参考答案求 max F1
#   Round 6 新增:
#     - 全 bug 修复 + multiref F1 评估 → R6-BUGFIX 作为真实 baseline
#     - 证据约束两步查询路径 → R6-EV / R6-EV-FULL
#     - refusal-flip stat, span_token_coverage 记录
#
# 实验矩阵 (4 卡并行, 卡 0-3):
#   GPU0  R6-BUGFIX:  evidence_mode=False  — Phase A fixes only; 真 multiref baseline
#   GPU1  R6-EV:      evidence_mode=True   — Phase B: 证据约束 (pre-gen verdict gate)
#   GPU2  R6-NOBLANK: evidence_mode=False  — 与 BUGFIX 同配置; 验证 acceptance gate 确定性
#   GPU3  R6-EV-FULL: evidence_mode=True   — Phase B: 证据约束 + post-gen repl 核查
#
# 注意: R6-NOBLANK 与 R6-BUGFIX 配置完全相同 (acceptance gate 在 evidence_mode=False 时
#   也始终激活: empty/length_bloat/negation_flip 三重检查); 两者 delta 应相同, 可做确定性核查。
#   R6-EV 与 R6-EV-FULL 同理 (post-gen repl 核查在 evidence_mode=True 时始终运行)。
#
# 用法: bash run_freeform_round6.sh [--limit N]  (默认 N=50)

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON=python3
MODEL_DIR=/dev/shm
OUTDIR="$REPO/outputs/freeform_grid/round6"
LOGDIR="$OUTDIR/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

LIMIT=50
for arg in "$@"; do
  case $arg in
    --limit=*) LIMIT="${arg#*=}" ;;
    --limit)   shift; LIMIT="$1"; shift ;;
  esac
done

echo "[round6] limit=$LIMIT  model=$MODEL_DIR/Qwen2.5-Omni-7B"
echo "[round6] Bug fixes: A1(marker strip) A2(char-offset) A3(multiref F1)"
echo "[round6] New metrics: span_token_coverage, refusal_flip_rate"

# 格式: "GPU  TAG  TAU  MAX_SPANS  EV_FLAG"
# 八卡全用: tau=0/0.3/0.5 × bugfix/evidence + max_spans=1 消融
declare -a EXPS=(
  "0  R6-BUGFIX        0    2  "
  "1  R6-EV            0    2  --evidence-mode"
  "2  R6-NOBLANK       0    2  "
  "3  R6-BUGFIX-MAX1   0    1  "
  "4  R6-BUGFIX-TAU03  0.3  2  "
  "5  R6-BUGFIX-TAU05  0.5  2  "
  "6  R6-EV-TAU03      0.3  2  --evidence-mode"
  "7  R6-EV-TAU05      0.5  2  --evidence-mode"
)

PIDS=()
TAGS=()

for spec in "${EXPS[@]}"; do
  read -r GPU TAG TAU MAX_SPANS EV_FLAG <<< "$spec"
  OUTFILE="$OUTDIR/haloquest_${TAG}.jsonl"
  LOGFILE="$LOGDIR/haloquest_${TAG}.log"

  CUDA_VISIBLE_DEVICES=$GPU \
  MMHALU_MODEL_ROOT=$MODEL_DIR \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/tmp/pip-target-b9a06nc9/lib/python \
  LD_LIBRARY_PATH=/dev/shm/so_mirror:/tmp/pip-target-b9a06nc9/lib/python/torch/lib \
    "$PYTHON" "$REPO/run_halo_cu121.py" \
      --tau "$TAU" \
      --limit "$LIMIT" \
      --min-span-len 1 \
      --max-spans "$MAX_SPANS" \
      --select-mode rho \
      --diag-mode none \
      $EV_FLAG \
      --out "$OUTFILE" \
    > "$LOGFILE" 2>&1 &

  PIDS+=($!)
  TAGS+=("$TAG")
  echo "[launch] GPU=$GPU  $TAG  tau=$TAU  max_spans=$MAX_SPANS  ev='$EV_FLAG'  pid=${PIDS[-1]}"
done

echo ""
echo "[round6] 全部 ${#PIDS[@]} 个进程已启动，等待完成..."
FAIL=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[done]  ${TAGS[$i]}  OK"
  else
    echo "[FAIL]  ${TAGS[$i]}  exit=$?"
    FAIL=$((FAIL+1))
  fi
done

echo ""
echo "===== ROUND 6 SUMMARY ====="
echo "failed=$FAIL / total=${#PIDS[@]}"
echo ""
printf "%-14s %-6s %-6s %-10s %-10s %-10s %-8s %-8s\n" \
  "tag" "ev" "delta" "base_f1" "rep_f1" "n_edited" "ref_flip" "cov_hi"
printf '%0.s-' {1..90}; echo ""

for spec in "${EXPS[@]}"; do
  read -r GPU TAG EV_FLAG <<< "$spec"
  F="$OUTDIR/haloquest_${TAG}.jsonl"
  if [ -f "$F" ]; then
    python3 - "$F" "$TAG" "$EV_FLAG" <<'PYEOF'
import sys, json
fpath, tag, ev_flag = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(fpath) if l.strip()]
ok = [r for r in rows if r.get("status") == "success"]
if ok:
    base  = sum(r["base_f1"] for r in ok) / len(ok)
    rep   = sum(r["repaired_f1"] for r in ok) / len(ok)
    delta = rep - base
    n_edited = sum(1 for r in ok if r.get("n_spans", 0) > 0 and not r.get("no_edit"))

    def _has_ref(t):
        tl = t.lower()
        for ph in ("there is no", "there are no", "cannot", "can't",
                   "don't", "doesn't", "isn't", "aren't", "wasn't", "weren't"):
            if ph in tl: return True
        return bool(set(tl.split()) & {"not","no","never","unable","unknown","unclear","undetermined"})

    n_ref_base = sum(1 for r in ok if _has_ref(r["base_pred"]))
    n_ref_flip = sum(1 for r in ok
                     if _has_ref(r["base_pred"]) and not _has_ref(r["repaired_pred"]))
    flip_rate = n_ref_flip / n_ref_base if n_ref_base else 0.0

    covs = [r.get("span_token_coverage", 0.0) for r in ok if r.get("n_spans", 0) > 0]
    frac_hi = sum(1 for c in covs if c > 0.8) / len(covs) if covs else 0.0
    mean_cov = sum(covs) / len(covs) if covs else 0.0

    ev_label = "Y" if "--evidence-mode" in ev_flag else "N"
    print(f"{tag:<14} {ev_label:<6} {delta:+.2f}  {base:<10.2f} {rep:<10.2f} "
          f"{n_edited:<8} {flip_rate:.2f}    {frac_hi:.2f}  "
          f"(mean_cov={mean_cov:.2f})")
else:
    print(f"{tag:<14} NO_OK_ROWS")
PYEOF
  else
    printf "%-14s %s\n" "$TAG" "OUTPUT MISSING"
  fi
done

echo ""
echo "===== CROSS-ROUND COMPARISON (rho, tau=0, max_spans=2) ====="
python3 - "$REPO/outputs/freeform_grid" "$OUTDIR" <<'PYEOF'
import sys, json, os
old_dir, new_dir = sys.argv[1], sys.argv[2]
history = [
    (os.path.join(old_dir, "haloquest_R2-C1.jsonl"),      "R2-C1     (single-ref, has bugs)"),
    (os.path.join(new_dir, "haloquest_R6-BUGFIX.jsonl"),  "R6-BUGFIX (multiref, bugs fixed)"),
    (os.path.join(new_dir, "haloquest_R6-EV.jsonl"),      "R6-EV     (evidence gate)"),
    (os.path.join(new_dir, "haloquest_R6-NOBLANK.jsonl"), "R6-NOBLANK(acceptance gate only)"),
    (os.path.join(new_dir, "haloquest_R6-EV-FULL.jsonl"), "R6-EV-FULL(ev+post-gen check)"),
]
results = []
for fpath, label in history:
    if not os.path.exists(fpath):
        continue
    rows = [json.loads(l) for l in open(fpath) if l.strip()]
    ok = [r for r in rows if r.get("status") == "success"]
    if ok:
        base = sum(r["base_f1"] for r in ok) / len(ok)
        rep  = sum(r["repaired_f1"] for r in ok) / len(ok)
        results.append((rep - base, label))

best = max(d for d, _ in results) if results else None
for delta, label in results:
    marker = " ← BEST" if delta == best else ""
    print(f"  {label:<45}  delta={delta:+.2f}{marker}")
PYEOF
