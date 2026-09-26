#!/usr/bin/env bash
# run_grid_parallel.sh — tau 网格并行驱动 (新文件, 不修改任何既有文件)
#
# 约束: 只能用卡 0-3 (卡 4-7 他人 sglang 进程, 严禁碰)
# tau 网格 8 个值, 4 张卡, 每卡串行跑 2 个 tau:
#   卡 0: tau=0,    tau=0.1
#   卡 1: tau=0.25, tau=0.5
#   卡 2: tau=1.0,  tau=2.0
#   卡 3: tau=4.0,  tau=+inf (inf 不需要 GPU, CPU 跑)
#
# 用法:
#   cd /home/tsq/rhq/omniconfess/OmniConfess
#   bash run_grid_parallel.sh [--limit 50]
#
# 输出:
#   outputs/freeform_grid/haloquest_tau{tau}.jsonl
#   /tmp/halo_grid_gpu{i}_tau{tau}.log

set -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/slow_share/tsq/venvs/omniconfess/bin/python"
SCRIPT="$REPO/run_halo_single.py"
OUTDIR="$REPO/outputs/freeform_grid"
LIMIT=50

# 解析参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUTDIR"

echo "[grid] Starting tau grid, limit=$LIMIT, $(date)"
echo "[grid] Output dir: $OUTDIR"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | head -4
echo ""

# 启动 4 个后台 worker, 每个 worker 串行跑 2 个 tau
# --- 卡 0: tau=0, tau=0.1 ---
(
  echo "[gpu0] Starting tau=0 on card 0, $(date)" | tee /tmp/halo_grid_gpu0_tau0.log
  CUDA_VISIBLE_DEVICES=0 MMHALU_MODEL_ROOT=/slow_share/tsq/MODELS HF_HUB_OFFLINE=1 \
    "$PYTHON" "$SCRIPT" --tau 0 --limit "$LIMIT" \
    --out "$OUTDIR/haloquest_tau0.jsonl" \
    >> /tmp/halo_grid_gpu0_tau0.log 2>&1
  echo "[gpu0] Done tau=0, rc=$?, $(date)" | tee -a /tmp/halo_grid_gpu0_tau0.log

  echo "[gpu0] Starting tau=0.1 on card 0, $(date)" | tee /tmp/halo_grid_gpu0_tau0.1.log
  CUDA_VISIBLE_DEVICES=0 MMHALU_MODEL_ROOT=/slow_share/tsq/MODELS HF_HUB_OFFLINE=1 \
    "$PYTHON" "$SCRIPT" --tau 0.1 --limit "$LIMIT" \
    --out "$OUTDIR/haloquest_tau0.1.jsonl" \
    >> /tmp/halo_grid_gpu0_tau0.1.log 2>&1
  echo "[gpu0] Done tau=0.1, rc=$?, $(date)" | tee -a /tmp/halo_grid_gpu0_tau0.1.log
) &
PID0=$!
echo "[grid] card 0 worker PID=$PID0 (tau=0, tau=0.1)"

# --- 卡 1: tau=0.25, tau=0.5 ---
(
  echo "[gpu1] Starting tau=0.25 on card 1, $(date)" | tee /tmp/halo_grid_gpu1_tau0.25.log
  CUDA_VISIBLE_DEVICES=1 MMHALU_MODEL_ROOT=/slow_share/tsq/MODELS HF_HUB_OFFLINE=1 \
    "$PYTHON" "$SCRIPT" --tau 0.25 --limit "$LIMIT" \
    --out "$OUTDIR/haloquest_tau0.25.jsonl" \
    >> /tmp/halo_grid_gpu1_tau0.25.log 2>&1
  echo "[gpu1] Done tau=0.25, rc=$?, $(date)" | tee -a /tmp/halo_grid_gpu1_tau0.25.log

  echo "[gpu1] Starting tau=0.5 on card 1, $(date)" | tee /tmp/halo_grid_gpu1_tau0.5.log
  CUDA_VISIBLE_DEVICES=1 MMHALU_MODEL_ROOT=/slow_share/tsq/MODELS HF_HUB_OFFLINE=1 \
    "$PYTHON" "$SCRIPT" --tau 0.5 --limit "$LIMIT" \
    --out "$OUTDIR/haloquest_tau0.5.jsonl" \
    >> /tmp/halo_grid_gpu1_tau0.5.log 2>&1
  echo "[gpu1] Done tau=0.5, rc=$?, $(date)" | tee -a /tmp/halo_grid_gpu1_tau0.5.log
) &
PID1=$!
echo "[grid] card 1 worker PID=$PID1 (tau=0.25, tau=0.5)"

# --- 卡 2: tau=1.0, tau=2.0 ---
(
  echo "[gpu2] Starting tau=1.0 on card 2, $(date)" | tee /tmp/halo_grid_gpu2_tau1.0.log
  CUDA_VISIBLE_DEVICES=2 MMHALU_MODEL_ROOT=/slow_share/tsq/MODELS HF_HUB_OFFLINE=1 \
    "$PYTHON" "$SCRIPT" --tau 1.0 --limit "$LIMIT" \
    --out "$OUTDIR/haloquest_tau1.0.jsonl" \
    >> /tmp/halo_grid_gpu2_tau1.0.log 2>&1
  echo "[gpu2] Done tau=1.0, rc=$?, $(date)" | tee -a /tmp/halo_grid_gpu2_tau1.0.log

  echo "[gpu2] Starting tau=2.0 on card 2, $(date)" | tee /tmp/halo_grid_gpu2_tau2.0.log
  CUDA_VISIBLE_DEVICES=2 MMHALU_MODEL_ROOT=/slow_share/tsq/MODELS HF_HUB_OFFLINE=1 \
    "$PYTHON" "$SCRIPT" --tau 2.0 --limit "$LIMIT" \
    --out "$OUTDIR/haloquest_tau2.0.jsonl" \
    >> /tmp/halo_grid_gpu2_tau2.0.log 2>&1
  echo "[gpu2] Done tau=2.0, rc=$?, $(date)" | tee -a /tmp/halo_grid_gpu2_tau2.0.log
) &
PID2=$!
echo "[grid] card 2 worker PID=$PID2 (tau=1.0, tau=2.0)"

# --- 卡 3: tau=4.0, tau=inf ---
(
  echo "[gpu3] Starting tau=4.0 on card 3, $(date)" | tee /tmp/halo_grid_gpu3_tau4.0.log
  CUDA_VISIBLE_DEVICES=3 MMHALU_MODEL_ROOT=/slow_share/tsq/MODELS HF_HUB_OFFLINE=1 \
    "$PYTHON" "$SCRIPT" --tau 4.0 --limit "$LIMIT" \
    --out "$OUTDIR/haloquest_tau4.0.jsonl" \
    >> /tmp/halo_grid_gpu3_tau4.0.log 2>&1
  echo "[gpu3] Done tau=4.0, rc=$?, $(date)" | tee -a /tmp/halo_grid_gpu3_tau4.0.log

  # tau=inf: CPU only (no GPU needed)
  echo "[gpu3] Starting tau=inf (CPU), $(date)" | tee /tmp/halo_grid_gpu3_tauinf.log
  CUDA_VISIBLE_DEVICES="" MMHALU_MODEL_ROOT=/slow_share/tsq/MODELS HF_HUB_OFFLINE=1 \
    "$PYTHON" "$SCRIPT" --tau 1e18 --limit "$LIMIT" \
    --out "$OUTDIR/haloquest_tauinf.jsonl" \
    >> /tmp/halo_grid_gpu3_tauinf.log 2>&1
  echo "[gpu3] Done tau=inf, rc=$?, $(date)" | tee -a /tmp/halo_grid_gpu3_tauinf.log
) &
PID3=$!
echo "[grid] card 3 worker PID=$PID3 (tau=4.0, tau=inf)"

echo ""
echo "[grid] All 4 workers launched. Waiting..."
echo "[grid] Monitor: tail -f /tmp/halo_grid_gpu*.log"
echo ""

# 等待所有 worker
wait $PID0; RC0=$?
echo "[grid] card 0 worker done, rc=$RC0, $(date)"
wait $PID1; RC1=$?
echo "[grid] card 1 worker done, rc=$RC1, $(date)"
wait $PID2; RC2=$?
echo "[grid] card 2 worker done, rc=$RC2, $(date)"
wait $PID3; RC3=$?
echo "[grid] card 3 worker done, rc=$RC3, $(date)"

echo ""
echo "[grid] All workers finished. Collecting results..."
echo "tau,n_run,n_success,n_regen_failed,n_samples_edited,base_tokenf1_mean,repaired_tokenf1_mean,delta_f1"
for f in "$OUTDIR"/haloquest_tau*.jsonl; do
  tau_tag=$(basename "$f" .jsonl | sed 's/haloquest_tau//')
  if [ -f "$f" ]; then
    python3 -c "
import json, sys
rows = [json.loads(l) for l in open('$f')]
n_run = len(rows)
n_ok = sum(1 for r in rows if r.get('status') == 'success')
n_fail = sum(1 for r in rows if r.get('status') == 'regen_failed')
n_edit = sum(1 for r in rows if r.get('n_spans',0) > 0 and r.get('status')=='success' and not r.get('no_edit'))
bf = [r['base_f1'] for r in rows if r.get('status')=='success']
rf = [r['repaired_f1'] for r in rows if r.get('status')=='success']
mb = sum(bf)/len(bf) if bf else 0
mr = sum(rf)/len(rf) if rf else 0
print(f'$tau_tag,{n_run},{n_ok},{n_fail},{n_edit},{mb:.2f},{mr:.2f},{mr-mb:.2f}')
" 2>/dev/null || echo "$tau_tag,ERROR"
  else
    echo "$tau_tag,NOT_FOUND"
  fi
done

echo ""
echo "[grid] Done, $(date)"
