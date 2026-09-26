#!/bin/bash
# run_freeform_round4.sh — Round 4: 诊断注入 prompt 的重生成 (diagnosis-aware regeneration)
#
# 背景:
#   Round 2 最佳: R2-C1 (tau=0, max_spans=2, rho)  delta=+0.88  — 但重生成 prompt 不含诊断
#   Round 3 失败: over_reliance 信号覆盖过广，under_reliance 信号过稀
#   根本问题: regenerate_spans 不知道每个 span 是"依赖不足"还是"依赖过度"
#
# Round 4 修复:
#   regenerate_spans 现在对每个 span 调用 span_dependency_diagnosis():
#     - 若 span 中多数 token 为 over_reliance (g3 高): 告知模型"你过度依赖了语言先验，
#       应忽略它，只根据视觉/音频证据重写"
#     - 若 span 中多数 token 为 under_reliance (g1 < 0): 告知模型"相关通道证据反对此内容，
#       说明你没有充分利用实际媒体信息，请仔细重读证据后改写"
#     - 并列出最强信号 token 及其 g1/g3 具体数值
#   该诊断文本作为 dep_diagnosis 注入 build_infill_prompt()
#
# 实验矩阵 (7 卡并行, 卡 0-6):
#   GPU0  R4-BASE:  rho    tau=0    max_spans=2  — 与R2-C1同配置, 直接验证诊断效果
#   GPU1  R4-OV:    over   tau=0    max_spans=2  — over_reliance 模式 + 诊断
#   GPU2  R4-OV-05: over   tau=0.5  max_spans=2  — 更精准的 over 信号 + 诊断
#   GPU3  R4-OV-10: over   tau=1.0  max_spans=2  — 高置信 over 信号 + 诊断
#   GPU4  R4-OV-20: over   tau=2.0  max_spans=2  — 极高置信 over + 诊断
#   GPU5  R4-EI:    either tau=0    max_spans=2  — 并集模式 + 诊断
#   GPU6  R4-UR:    under  tau=0    max_spans=2  — under_reliance 模式 + 诊断
#
# 关键对比:
#   R4-BASE  vs  R2-C1  (+0.88): 验证诊断有无单独价值
#   R4-OV-*  vs  R3-OV-*:       验证诊断能否弥补 over 信号过广的问题
#   R4-UR    vs  R3-UR-05:       验证诊断对稀疏 under 信号的帮助
#
# 用法: bash run_freeform_round4.sh [--limit N]  (默认 N=50)

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON=python3
MODEL_DIR=/dev/shm
OUTDIR="$REPO/outputs/freeform_grid"
LOGDIR="$OUTDIR/logs_r4"
mkdir -p "$OUTDIR" "$LOGDIR"

LIMIT=50
for arg in "$@"; do
  case $arg in
    --limit=*) LIMIT="${arg#*=}" ;;
    --limit)   shift; LIMIT="$1"; shift ;;
  esac
done

echo "[round4] limit=$LIMIT  model=$MODEL_DIR/Qwen2.5-Omni-7B"
echo "[round4] 本轮新增: span_dependency_diagnosis() 注入重生成 prompt"

# (GPU, tau, min_span, max_spans, select_mode, tag)
declare -a EXPS=(
  "0  0    1  2  rho            R4-BASE"
  "1  0    1  2  over_reliance  R4-OV"
  "2  0.5  1  2  over_reliance  R4-OV-05"
  "3  1.0  1  2  over_reliance  R4-OV-10"
  "4  2.0  1  2  over_reliance  R4-OV-20"
  "5  0    1  2  either         R4-EI"
  "6  0    1  2  under_reliance R4-UR"
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
echo "[round4] 全部 ${#PIDS[@]} 个进程已启动，等待完成..."
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
echo "===== ROUND 4 SUMMARY (diagnosis-aware regeneration) ====="
echo "failed=$FAIL / total=${#PIDS[@]}"
echo ""
printf "%-12s %-16s %-6s %-10s %-10s %-10s %-10s\n" \
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
edited = [r for r in ok if r.get("n_spans", 0) > 0 and not r.get("no_edit")]
if ok:
    base  = sum(r["base_f1"] for r in ok) / len(ok)
    rep   = sum(r["repaired_f1"] for r in ok) / len(ok)
    delta = rep - base
    # 诊断注入情况: 统计 replacements 中有 dep_diagnosis 的比例
    n_with_diag = sum(
        1 for r in ok
        for rpl in r.get("replacements", [])
        if rpl.get("dep_diagnosis")
    )
    n_total_rpl = sum(len(r.get("replacements", [])) for r in ok)
    diag_pct = 100.0 * n_with_diag / n_total_rpl if n_total_rpl else 0.0
    print(f"{tag:<12} {mode:<16} {tau:<6} {ms:<10} "
          f"{base:<10.2f} {rep:<10.2f} {delta:+.2f}  "
          f"(edited={len(edited)}, diag={diag_pct:.0f}%)")
else:
    print(f"{tag:<12} {mode:<16} {tau:<6} {ms:<10} NO_OK_ROWS")
PYEOF
  else
    printf "%-12s %-16s %-6s %-10s %s\n" "$TAG" "$MODE" "$TAU" "$MAX_SPANS" "OUTPUT MISSING"
  fi
done

echo ""
echo "===== CROSS-ROUND COMPARISON ====="
echo "  R2-C1  (no-diag, rho,   tau=0,  max2): baseline +0.88"
echo "  R3-OV-05-C (no-diag, over, tau=0.5, max2): see round3 results"
python3 - "$OUTDIR" <<'PYEOF'
import sys, json, os
d = sys.argv[1]
rows_map = {}
for tag in ["R2-C1", "R3-OV-05-C", "R4-BASE", "R4-OV", "R4-OV-05", "R4-OV-10", "R4-EI", "R4-UR"]:
    f = os.path.join(d, f"haloquest_{tag}.jsonl")
    if not os.path.exists(f):
        continue
    rows = [json.loads(l) for l in open(f) if l.strip()]
    ok = [r for r in rows if r.get("status") == "success"]
    if ok:
        base  = sum(r["base_f1"] for r in ok) / len(ok)
        rep   = sum(r["repaired_f1"] for r in ok) / len(ok)
        rows_map[tag] = rep - base
for tag, delta in rows_map.items():
    marker = " ← best" if delta == max(rows_map.values()) else ""
    print(f"  {tag:<16}  delta={delta:+.2f}{marker}")
PYEOF
