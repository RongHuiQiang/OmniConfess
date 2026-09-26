#!/bin/bash
# run_freeform_round3.sh — Round 3 依赖类型感知 span 选择
#
# Round 2 诊断:
#   - tau=0 时 88% token 被标为幻觉，相当于重写整个答案 → 全负
#   - under_reliance (-g1): 7.3%，信号小 (p50=0.11)
#   - over_reliance (g3):   67.9%，信号强 (p50=0.58，最大 19.9)
#   - 真正的幻觉集中在高 g3 的少数 token，用 tau=0 把大量低 g3 噪声也圈进来了
#
# Round 3 策略:
#   - 用 select_mode=over_reliance (g3>tau) 代替 rho>tau
#   - 用 select_mode=under_reliance (-g1>tau) 单独测
#   - 用 select_mode=either 测并集
#   - tau 用有意义的阈值 (0.5 / 1.0 / 2.0)
#   - 配合 max_spans=2 控制修订量
#
# 实验矩阵 (7 卡并行，卡 0-6):
#   GPU0  R3-OV-05:  over_reliance   tau=0.5   max_spans=0
#   GPU1  R3-OV-10:  over_reliance   tau=1.0   max_spans=0
#   GPU2  R3-OV-05-C: over_reliance  tau=0.5   max_spans=2
#   GPU3  R3-OV-10-C: over_reliance  tau=1.0   max_spans=2
#   GPU4  R3-UR-05:  under_reliance  tau=0.5   max_spans=0
#   GPU5  R3-EI-05:  either          tau=0.5   max_spans=2
#   GPU6  R3-OV-20:  over_reliance   tau=2.0   max_spans=0
#
# 用法: bash run_freeform_round3.sh [--limit N]  (默认 N=50)

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON=python3
MODEL_DIR=/dev/shm
OUTDIR="$REPO/outputs/freeform_grid"
LOGDIR="$OUTDIR/logs_r3"
mkdir -p "$OUTDIR" "$LOGDIR"

LIMIT=50
for arg in "$@"; do
  case $arg in
    --limit=*) LIMIT="${arg#*=}" ;;
    --limit)   shift; LIMIT="$1"; shift ;;
  esac
done

echo "[round3] limit=$LIMIT  model=$MODEL_DIR/Qwen2.5-Omni-7B"

# (GPU, tau, min_span, max_spans, select_mode, tag)
declare -a EXPS=(
  "0  0.5  1  0  over_reliance  R3-OV-05"
  "1  1.0  1  0  over_reliance  R3-OV-10"
  "2  0.5  1  2  over_reliance  R3-OV-05-C"
  "3  1.0  1  2  over_reliance  R3-OV-10-C"
  "4  0.5  1  0  under_reliance R3-UR-05"
  "5  0.5  1  2  either         R3-EI-05"
  "6  2.0  1  0  over_reliance  R3-OV-20"
)

PIDS=()
TAGS=()

for spec in "${EXPS[@]}"; do
  read -r GPU TAU MIN_SPAN MAX_SPANS MODE TAG <<< "$spec"
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
      --min-span-len "$MIN_SPAN" \
      --max-spans "$MAX_SPANS" \
      --select-mode "$MODE" \
      --out "$OUTFILE" \
    > "$LOGFILE" 2>&1 &

  PIDS+=($!)
  TAGS+=("$TAG")
  echo "[launch] GPU=$GPU  $TAG  mode=$MODE  tau=$TAU  max_spans=$MAX_SPANS  pid=${PIDS[-1]}"
done

echo ""
echo "[round3] 全部 ${#PIDS[@]} 个进程已启动，等待完成..."
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
echo "===== ROUND 3 SUMMARY ====="
echo "failed=$FAIL / total=${#PIDS[@]}"
echo ""
printf "%-14s %-14s %-6s %-10s %-10s %-10s %-10s\n" \
  "tag" "mode" "tau" "max_spans" "base_f1" "repair_f1" "delta_f1"
printf '%0.s-' {1..80}; echo ""

for spec in "${EXPS[@]}"; do
  read -r GPU TAU MIN_SPAN MAX_SPANS MODE TAG <<< "$spec"
  F="$OUTDIR/haloquest_${TAG}.jsonl"
  if [ -f "$F" ]; then
    python3 - "$F" "$TAG" "$MODE" "$TAU" "$MAX_SPANS" <<'PYEOF'
import sys, json
fpath, tag, mode, tau, ms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
rows = [json.loads(l) for l in open(fpath) if l.strip()]
ok = [r for r in rows if r.get("status") == "success"]
edited = [r for r in ok if r.get("n_spans",0) > 0 and not r.get("no_edit")]
if ok:
    base  = sum(r["base_f1"] for r in ok) / len(ok)
    rep   = sum(r["repaired_f1"] for r in ok) / len(ok)
    delta = rep - base
    print(f"{tag:<14} {mode:<14} {tau:<6} {ms:<10} {base:<10.2f} {rep:<10.2f} {delta:+.2f}  (edited={len(edited)})")
else:
    print(f"{tag:<14} {mode:<14} {tau:<6} {ms:<10} NO_OK_ROWS")
PYEOF
  else
    printf "%-14s %-14s %-6s %-10s %s\n" "$TAG" "$MODE" "$TAU" "$MAX_SPANS" "OUTPUT MISSING"
  fi
done
