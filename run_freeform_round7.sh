#!/bin/bash
# run_freeform_round7.sh — Round 7: Coverage-Gated Repair
#
# 背景:
#   Round 6 (bugfix + multiref F1) 真实 delta 全负 (最优 -2.11)。
#   根因: tau=0 下 rho 把描述类回答几乎所有 token 标为待修订，
#   Top-2 span 合并后覆盖 89% answer token，等于整句重写。
#
#   Round 7 用两种机制解决覆盖率问题:
#     A) coverage_cap: 覆盖率 > cap 直接跳过，只编辑真正局部的 span
#     B) topk select_mode: top-2 rho_i token + ±2 token 窗口，强制窄 span
#     C) 高 tau 消融: tau=1.0/2.0 是否能自然产生局部 span
#
#   关键诊断量: delta_edited_only (只在实际被编辑的样本上计算 delta)
#   如果 delta_edited_only > 0，说明覆盖率 cap 有效，局部修复质量可以为正
#
# 实验矩阵 (8 卡):
#   GPU  TAG             TAU  MAX_SPANS  COV_CAP  SELECT
#   0    R7-COV60        0    2          0.60     rho
#   1    R7-COV40        0    2          0.40     rho
#   2    R7-COV20        0    2          0.20     rho
#   3    R7-MAX1-COV60   0    1          0.60     rho
#   4    R7-TAU10        1.0  2          0        rho
#   5    R7-TAU20        2.0  2          0        rho
#   6    R7-MAX1-TAU10   1.0  1          0        rho
#   7    R7-TOPK         0    2          0        topk
#
# 用法: bash run_freeform_round7.sh [--limit N]  (默认 N=50)

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON=python3
MODEL_DIR=/dev/shm
OUTDIR="$REPO/outputs/freeform_grid/round7"
LOGDIR="$OUTDIR/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

LIMIT=50
for arg in "$@"; do
  case $arg in
    --limit=*) LIMIT="${arg#*=}" ;;
    --limit)   shift; LIMIT="$1"; shift ;;
  esac
done

echo "[round7] limit=$LIMIT  model=$MODEL_DIR/Qwen2.5-Omni-7B"
echo "[round7] coverage-gated repair: coverage_cap + topk select_mode"

# 格式: "GPU  TAG  TAU  MAX_SPANS  COV_CAP  SELECT_MODE"
declare -a EXPS=(
  "0  R7-COV60       0    2  0.60  rho"
  "1  R7-COV40       0    2  0.40  rho"
  "2  R7-COV20       0    2  0.20  rho"
  "3  R7-MAX1-COV60  0    1  0.60  rho"
  "4  R7-TAU10       1.0  2  0     rho"
  "5  R7-TAU20       2.0  2  0     rho"
  "6  R7-MAX1-TAU10  1.0  1  0     rho"
  "7  R7-TOPK        0    2  0     topk"
)

PIDS=()
TAGS=()

for spec in "${EXPS[@]}"; do
  read -r GPU TAG TAU MAX_SPANS COV_CAP SELECT_MODE <<< "$spec"
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
      --select-mode "$SELECT_MODE" \
      --diag-mode none \
      --coverage-cap "$COV_CAP" \
      --out "$OUTFILE" \
    > "$LOGFILE" 2>&1 &

  PIDS+=($!)
  TAGS+=("$TAG")
  echo "[launch] GPU=$GPU  $TAG  tau=$TAU  max_spans=$MAX_SPANS  cov_cap=$COV_CAP  select=$SELECT_MODE  pid=${PIDS[-1]}"
done

echo ""
echo "[round7] 全部 ${#PIDS[@]} 个进程已启动，等待完成..."
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
echo "===== ROUND 7 SUMMARY ====="
echo "failed=$FAIL / total=${#PIDS[@]}"
echo ""
printf "%-16s %-5s %-5s %-6s %-8s %-10s %-10s %-8s %-8s %-8s\n" \
  "tag" "tau" "cap" "delta" "d_edited" "base_f1" "rep_f1" "cov_skip%" "n_edited" "mean_cov"
printf '%0.s-' {1..100}; echo ""

for spec in "${EXPS[@]}"; do
  read -r GPU TAG TAU MAX_SPANS COV_CAP SELECT_MODE <<< "$spec"
  F="$OUTDIR/haloquest_${TAG}.jsonl"
  if [ -f "$F" ]; then
    python3 - "$F" "$TAG" "$TAU" "$COV_CAP" "$SELECT_MODE" <<'PYEOF'
import sys, json
fpath, tag, tau, cov_cap, select_mode = sys.argv[1:]
rows = [json.loads(l) for l in open(fpath) if l.strip()]
ok = [r for r in rows if r.get("status") == "success"]
if ok:
    base  = sum(r["base_f1"]     for r in ok) / len(ok)
    rep   = sum(r["repaired_f1"] for r in ok) / len(ok)
    delta = rep - base

    edited = [r for r in ok if r.get("n_spans", 0) > 0]
    n_skip = sum(1 for r in edited if r.get("coverage_cap_skip", False))
    frac_skip = n_skip / len(edited) if edited else 0.0

    actually_edited = [r for r in edited if not r.get("coverage_cap_skip", False)]
    d_edited = (
        sum(r["repaired_f1"] - r["base_f1"] for r in actually_edited) / len(actually_edited)
        if actually_edited else 0.0
    )

    covs = [r.get("span_token_coverage", 0.0) for r in edited]
    mean_cov = sum(covs) / len(covs) if covs else 0.0

    print(f"{tag:<16} {tau:<5} {cov_cap:<5} {delta:+6.2f}  {d_edited:+7.2f}  "
          f"{base:<10.2f} {rep:<10.2f} {frac_skip*100:6.1f}%   "
          f"{len(actually_edited):<8} {mean_cov:.2f}")
else:
    print(f"{tag:<16} NO_OK_ROWS")
PYEOF
  else
    printf "%-16s %s\n" "$TAG" "OUTPUT MISSING"
  fi
done

echo ""
echo "===== CROSS-ROUND COMPARISON (tau=0, rho, max_spans=2, no cap) ====="
python3 - "$REPO/outputs/freeform_grid" "$OUTDIR" <<'PYEOF'
import sys, json, os
old_dir, new_dir = sys.argv[1], sys.argv[2]
history = [
    (os.path.join(old_dir, "haloquest_R2-C1.jsonl"),          "R2-C1          (single-ref, buggy eval)"),
    (os.path.join(old_dir, "round6", "haloquest_R6-BUGFIX.jsonl"), "R6-BUGFIX      (multiref, bugs fixed)"),
    (os.path.join(new_dir, "haloquest_R7-COV60.jsonl"),        "R7-COV60       (coverage_cap=0.60)"),
    (os.path.join(new_dir, "haloquest_R7-COV40.jsonl"),        "R7-COV40       (coverage_cap=0.40)"),
    (os.path.join(new_dir, "haloquest_R7-COV20.jsonl"),        "R7-COV20       (coverage_cap=0.20)"),
    (os.path.join(new_dir, "haloquest_R7-MAX1-COV60.jsonl"),   "R7-MAX1-COV60  (max_spans=1+cap=0.60)"),
    (os.path.join(new_dir, "haloquest_R7-TAU10.jsonl"),        "R7-TAU10       (tau=1.0)"),
    (os.path.join(new_dir, "haloquest_R7-TAU20.jsonl"),        "R7-TAU20       (tau=2.0)"),
    (os.path.join(new_dir, "haloquest_R7-MAX1-TAU10.jsonl"),   "R7-MAX1-TAU10  (tau=1.0+max_spans=1)"),
    (os.path.join(new_dir, "haloquest_R7-TOPK.jsonl"),         "R7-TOPK        (topk select_mode)"),
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
