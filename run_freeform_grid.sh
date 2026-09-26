#!/bin/bash
# run_freeform_grid.sh — 8 卡并行跑 haloquest tau 网格
# 关键: MMHALU_MODEL_ROOT=/dev/shm 让 load_model() 从 shm 加载权重,绕开 NFS mmap 死锁
# τ 值: 0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, +inf(99999) 各占一张卡 (卡 0-7)
# 用法: bash run_freeform_grid.sh [--limit N]  (默认 N=50; 0=全量)

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/slow_share/tsq/venvs/omniconfess/bin/python
MODEL_DIR=/dev/shm   # Qwen2.5-Omni-7B 已在 /dev/shm/Qwen2.5-Omni-7B (21G, 完整)
OUTDIR="$REPO/outputs/freeform_grid"
LOGDIR="$OUTDIR/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

LIMIT=50
for arg in "$@"; do
  case $arg in
    --limit) shift; LIMIT="$1"; shift ;;
    --limit=*) LIMIT="${arg#*=}" ;;
  esac
done

TAUS=(0 0.1 0.25 0.5 1.0 2.0 4.0 99999)
DATASET=haloquest

echo "[grid] 启动 8 卡并行, dataset=$DATASET limit=$LIMIT"
echo "[grid] 模型路径: $MODEL_DIR/Qwen2.5-Omni-7B"

PIDS=()
for i in "${!TAUS[@]}"; do
  TAU="${TAUS[$i]}"
  GPU=$i
  OUTFILE="$OUTDIR/${DATASET}_tau${TAU}_gpu${GPU}.jsonl"
  LOGFILE="$LOGDIR/${DATASET}_tau${TAU}_gpu${GPU}.log"

  CUDA_VISIBLE_DEVICES=$GPU \
  MMHALU_MODEL_ROOT=$MODEL_DIR \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
    "$PYTHON" "$REPO/run_haloquest_pilot.py" \
      --limit "$LIMIT" \
      --tau "$TAU" \
      --prior-baseline none \
      --out "$OUTFILE" \
    > "$LOGFILE" 2>&1 &

  PIDS+=($!)
  echo "[grid] GPU $GPU  tau=$TAU  pid=${PIDS[$i]}  log=$LOGFILE"
done

echo "[grid] 全部 ${#PIDS[@]} 个进程已启动, 等待完成..."
FAIL=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[done] GPU $i  tau=${TAUS[$i]}  OK"
  else
    echo "[FAIL] GPU $i  tau=${TAUS[$i]}  exit=$?"
    FAIL=$((FAIL+1))
  fi
done

echo ""
echo "===== GRID SUMMARY ====="
echo "failed=$FAIL / total=${#PIDS[@]}"
for i in "${!TAUS[@]}"; do
  F="$OUTDIR/${DATASET}_tau${TAUS[$i]}_gpu${i}.jsonl"
  if [ -f "$F" ]; then
    N=$(wc -l < "$F")
    # 取最后一行的 summary 统计
    SUMMARY=$(tail -1 "$F" 2>/dev/null | python3 -c \
      "import sys,json; d=json.loads(sys.stdin.read()); \
       print(f\"n={d.get('n_run','?')} base_f1={d.get('base_tokenf1_mean','?')} repair_f1={d.get('repaired_tokenf1_mean','?')} delta={d.get('delta_f1','?')}\")" 2>/dev/null || echo "n=$N lines")
    echo "  tau=${TAUS[$i]}  $SUMMARY"
  else
    echo "  tau=${TAUS[$i]}  OUTPUT MISSING"
  fi
done
