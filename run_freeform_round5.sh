#!/bin/bash
# run_freeform_round5.sh — Round 5: 软诊断注入 (soft hint, 不暴露机制语言)
#
# Round 4 诊断:
#   hard diag (g1/g3 数值/通道名) 全面反效果 (-1.14 vs R2-C1 +0.88)
#   根因: 模型无法将"g3=2.3, 语言先验驱动"等机制语言映射为正确修订动作
#   反而引导模型生成听起来更"视觉化"但实际更不准确的答案
#
# Round 5 策略: diag_mode=soft
#   不暴露任何内部机制 (无 g1/g3/通道名/token 计数)
#   只给操作性自然语言指令:
#     over_reliance → "Look carefully at the image again and describe only what
#                      you can directly observe — do not infer or assume."
#     under_reliance → "This span appears to contradict what the image shows.
#                       Re-examine and correct it."
#   对比实验: 同一 select_mode/tau/max_spans 配置, 只改 diag_mode none→soft
#
# 实验矩阵 (7 卡并行, 卡 0-6):
#   GPU0  R5-BASE:   rho   tau=0  max_spans=2  soft  — vs R2-C1 (同配置无diag, +0.88)
#   GPU1  R5-OV:     over  tau=0  max_spans=2  soft  — vs R4-OV  (-1.73 hard)
#   GPU2  R5-OV-05:  over  tau=0.5 max_spans=2 soft  — vs R4-OV-05 (-4.52 hard)
#   GPU3  R5-OV-10:  over  tau=1.0 max_spans=2 soft  — vs R4-OV-10 (-4.80 hard)
#   GPU4  R5-OV-20:  over  tau=2.0 max_spans=2 soft  — vs R4-OV-20 (-5.96 hard)
#   GPU5  R5-EI:     either tau=0 max_spans=2  soft  — vs R4-EI   (-1.14 hard)
#   GPU6  R5-UR:     under tau=0  max_spans=2  soft  — vs R4-UR   (-2.68 hard)
#
# 用法: bash run_freeform_round5.sh [--limit N]  (默认 N=50)

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON=python3
MODEL_DIR=/dev/shm
OUTDIR="$REPO/outputs/freeform_grid"
LOGDIR="$OUTDIR/logs_r5"
mkdir -p "$OUTDIR" "$LOGDIR"

LIMIT=50
for arg in "$@"; do
  case $arg in
    --limit=*) LIMIT="${arg#*=}" ;;
    --limit)   shift; LIMIT="$1"; shift ;;
  esac
done

echo "[round5] limit=$LIMIT  model=$MODEL_DIR/Qwen2.5-Omni-7B"
echo "[round5] diag_mode=soft (操作性指令, 不暴露 g1/g3 机制语言)"

# (GPU, tau, min_span, max_spans, select_mode, tag)
declare -a EXPS=(
  "0  0    1  2  rho            R5-BASE"
  "1  0    1  2  over_reliance  R5-OV"
  "2  0.5  1  2  over_reliance  R5-OV-05"
  "3  1.0  1  2  over_reliance  R5-OV-10"
  "4  2.0  1  2  over_reliance  R5-OV-20"
  "5  0    1  2  either         R5-EI"
  "6  0    1  2  under_reliance R5-UR"
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
      --diag-mode soft \
      --out "$OUTFILE" \
    > "$LOGFILE" 2>&1 &

  PIDS+=($!)
  TAGS+=("$TAG")
  echo "[launch] GPU=$GPU  $TAG  mode=$MODE  tau=$TAU  max_spans=$MAX_SPANS  pid=${PIDS[-1]}"
done

echo ""
echo "[round5] 全部 ${#PIDS[@]} 个进程已启动，等待完成..."
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
echo "===== ROUND 5 SUMMARY (soft hint) ====="
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
echo "===== CROSS-ROUND COMPARISON (rho, tau=0, max_spans=2) ====="
python3 - "$OUTDIR" <<'PYEOF'
import sys, json, os
d = sys.argv[1]
# 全部历史 BASE 对比点
history = [
    ("R2-C1",   "no-diag"),
    ("R4-BASE",  "hard-diag"),
    ("R5-BASE",  "soft-diag"),
]
best_delta = -999
best_tag = ""
rows_map = {}
for tag, label in history:
    f = os.path.join(d, f"haloquest_{tag}.jsonl")
    if not os.path.exists(f):
        continue
    rows = [json.loads(l) for l in open(f) if l.strip()]
    ok = [r for r in rows if r.get("status") == "success"]
    if ok:
        base  = sum(r["base_f1"] for r in ok) / len(ok)
        rep   = sum(r["repaired_f1"] for r in ok) / len(ok)
        delta = rep - base
        rows_map[tag] = (delta, label)
        if delta > best_delta:
            best_delta = delta
            best_tag = tag

for tag, (delta, label) in rows_map.items():
    marker = " ← best" if tag == best_tag else ""
    print(f"  {tag:<10} ({label:<10})  delta={delta:+.2f}{marker}")

# 全面对比表
print("")
print("===== ALL ROUNDS FULL COMPARISON =====")
all_tags = [
    "R2-C1", "R3-OV-05-C",
    "R4-BASE", "R4-OV", "R4-OV-05", "R4-OV-10", "R4-OV-20", "R4-EI", "R4-UR",
    "R5-BASE", "R5-OV", "R5-OV-05", "R5-OV-10", "R5-OV-20", "R5-EI", "R5-UR",
]
all_results = []
for tag in all_tags:
    f = os.path.join(d, f"haloquest_{tag}.jsonl")
    if not os.path.exists(f):
        continue
    rows = [json.loads(l) for l in open(f) if l.strip()]
    ok = [r for r in rows if r.get("status") == "success"]
    if ok:
        base  = sum(r["base_f1"] for r in ok) / len(ok)
        rep   = sum(r["repaired_f1"] for r in ok) / len(ok)
        all_results.append((rep - base, tag))

all_results.sort(reverse=True)
for delta, tag in all_results:
    marker = " ← BEST" if tag == all_results[0][1] else ""
    print(f"  {tag:<16}  delta={delta:+.2f}{marker}")
PYEOF
