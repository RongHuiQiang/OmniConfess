#!/bin/bash
# run_grid_supply.sh -- persistent baseline-grid supply daemon.
# Each round: reread probe lock -> compute available cards -> rescan disk for
# incomplete LAUNCHABLE cells -> for each free available card launch the next
# cell (one-per-card, atomic mkdir claim) -> sleep. Crash/restart safe: no state
# is cached; the queue is recomputed from disk every round.
#
# Cards: BASE_CARDS={0,1,3}. card2 usable ONLY when /tmp/rhq_probe_card.lock is
# absent (reread every round, never cached). card3 hosts a self-managed nemotron
# SC loop -> this daemon never launches on card3.
# Only cells with an existing runner are launchable: Qwen3 {cad,dola,gd,pd,sc,tot,vcd}.
# All families tot/gd/pd = 1/3 (via *_1of3.json); opera = 1/3 on description
# family, FULL on judgement family; avcd and the rest FULL.
set -u
cd /home/tsq/rhq/omniconfess/OmniConfess
LOCK=/tmp/rhq_probe_card.lock
CLAIMDIR=/tmp/grid_supply_claims
mkdir -p "$CLAIMDIR"
LOG_PREFIX="[supply]"
POLL=90

MEM_FREE_MIB=2000

# expected full rows
declare -A EXP_FULL=( [pubmedqa]=468 [cmm]=721 [phd]=906 [ragtruth]=435 [haloquest]=365 [avhbench]=645 )
# 1/3 rows: judgement family {pubmedqa,cmm,phd} added 2026-08-22 (grid slimming)
declare -A EXP_13=(   [pubmedqa]=156 [cmm]=240 [phd]=302 [ragtruth]=145 [haloquest]=122 [avhbench]=215 )

JUDGE="pubmedqa cmm phd"
DESC="ragtruth haloquest avhbench"
QWEN3_METHODS="cad dola gd pd sc tot vcd avcd opera"   # avcd/opera runners ported+smoked 2026-08-21

# echo "ds method data exp" for each INCOMPLETE launchable cell, one per line
compute_queue() {
  for ds in $JUDGE $DESC; do
    for m in $QWEN3_METHODS; do
      local exp data
      if echo "tot gd pd" | grep -qw "$m"; then
        # degenerate cell (any family): 1/3
        data="OmniHalluBench/${ds}/${ds}_1of3.json"; exp=${EXP_13[$ds]}
      elif [ "$m" = "opera" ] && echo " $DESC " | grep -q " $ds "; then
        # opera on description family: 1/3; opera on judgement family stays FULL
        data="OmniHalluBench/${ds}/${ds}_1of3.json"; exp=${EXP_13[$ds]}
      else
        data="OmniHalluBench/${ds}/${ds}.json"; exp=${EXP_FULL[$ds]}
      fi
      [ -f "$data" ] || continue
      local cell="${ds}_qwen3_${m}"
      local out="outputs/${cell}"
      # completeness: scored file exists with rows >= exp (cell_done.txt is a
      # bonus stamp but not required -- some cells were completed by hand).
      local scored="$out/${cell}_full_single_scored.jsonl"
      if [ -f "$scored" ]; then
        local sn; sn=$(wc -l < "$scored" 2>/dev/null || echo 0)
        [ "$sn" -ge "$exp" ] && continue
      fi
      # skip if already claimed/running by this daemon
      [ -d "$CLAIMDIR/${cell}.run" ] && continue
      echo "$ds $m $data $exp"
    done
  done
}

card_free() {  # card_free <gpu> -> 0 if mem<MEM_FREE_MIB
  local g=$1 used
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  [ -n "$used" ] && [ "$used" -lt "$MEM_FREE_MIB" ]
}

available_cards() {  # cards 0-3 ours; 4-7 RETURNED 2026-08-22; cards listed in /tmp/grid_card_reserve are reserved for ablation/probe agents
  local c cards=""
  for c in 0 1 2 3 4 5 6 7; do
    grep -qw "$c" /tmp/grid_card_reserve 2>/dev/null && continue
    cards="$cards $c"
  done
  echo "$cards"
}

launch_cell() {  # launch_cell <gpu> <ds> <method> <data> <exp>
  local g=$1 ds=$2 m=$3 data=$4 exp=$5
  local cell="${ds}_qwen3_${m}"
  local claim="$CLAIMDIR/${cell}.run"
  # guard: never launch a cell whose output_dir is already in a LIVE process
  # cmdline (covers external orphan runs on card0/card1 not under our claim).
  if pgrep -af "outputs/${cell}[ /]" >/dev/null 2>&1; then
    echo "$LOG_PREFIX $(date '+%F %T') SKIP $cell -- live external process on it"
    return 1
  fi
  # atomic claim: also acts as card busy for the duration
  mkdir "$claim" 2>/dev/null || return 1
  echo "$g $(date +%s)" > "$claim/info"
  echo "$LOG_PREFIX $(date '+%F %T') LAUNCH $cell gpu=$g exp=$exp data=$data"
  mkdir -p "outputs/${cell}"
  (
    setsid bash run_qwen3_grid_cell.sh "$g" "$ds" "$m" "$data" "$exp" \
      </dev/null >>"outputs/${cell}/run.log" 2>&1
    rc=$?
    echo "$LOG_PREFIX $(date '+%F %T') DONE  $cell gpu=$g rc=$rc" >> outputs/grid_supply.log
    rm -rf "$claim"
  ) &
  return 0
}

# card busy according to THIS daemon = any active claim whose info card==g
card_busy_by_daemon() {
  local g=$1 f
  for f in "$CLAIMDIR"/*.run/info; do
    [ -f "$f" ] || continue
    local cg; cg=$(awk '{print $1}' "$f")
    [ "$cg" = "$g" ] && return 0
  done
  return 1
}

echo "$LOG_PREFIX $(date '+%F %T') supply daemon start pid=$$"
while true; do
  # orphan-claim GC: a prior daemon may have died leaving a .run whose cell
  # process is no longer alive (its self-rm never fired). If no live process
  # references the cell's output dir, drop the claim so the cell re-enters the
  # queue. Claims for cells with a live process are kept (real, in-flight).
  for cf in "$CLAIMDIR"/*.run; do
    [ -d "$cf" ] || continue
    ccell=$(basename "$cf" .run)
    if ! pgrep -af "outputs/${ccell}[ /]" >/dev/null 2>&1; then
      echo "$LOG_PREFIX $(date '+%F %T') GC orphan claim $ccell (no live process)"
      rm -rf "$cf"
    fi
  done
  mapfile -t QUEUE < <(compute_queue)
  navail=0
  for g in $(available_cards); do
    card_busy_by_daemon "$g" && continue
    card_free "$g" || continue
    navail=$((navail+1))
    # pop first queue item not already claimed
    launched=0
    for spec in "${QUEUE[@]}"; do
      [ -z "$spec" ] && continue
      read -r ds m data exp <<< "$spec"
      cell="${ds}_qwen3_${m}"
      [ -d "$CLAIMDIR/${cell}.run" ] && continue
      if launch_cell "$g" "$ds" "$m" "$data" "$exp"; then launched=1; break; fi
    done
    [ "$launched" = "1" ] && sleep 20   # stagger model loads
  done
  nq=${#QUEUE[@]}
  echo "$LOG_PREFIX $(date '+%F %T') round: lock=$([ -f "$LOCK" ] && cat "$LOCK" || echo none) avail_cards=[$(available_cards)] incomplete=$nq"
  [ "$nq" -eq 0 ] && echo "$LOG_PREFIX $(date '+%F %T') queue empty; idle-polling"
  sleep "$POLL"
done
