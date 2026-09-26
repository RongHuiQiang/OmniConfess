#!/bin/bash
# pd_watch.sh -- persistent setsid watchdog for cmm_qwen3_pd grid cell.
# Polls grid_supply.log for fresh LAUNCH lines of pd cells; after a settle
# delay dumps run.log head/tail to /tmp/pd_watch2.log; if the log shows the
# known int/list crash signature, pkills that cell so the supply daemon
# relaunches it.
set -u
cd /home/tsq/rhq/omniconfess/OmniConfess
SUPPLY=outputs/grid_supply.log
OUT=/tmp/pd_watch2.log
BASE=$(wc -l < "$SUPPLY" 2>/dev/null || echo 0)

log() { echo "[pd_watch $(date '+%F %T')] $*" >> "$OUT"; }
log "watchdog start pid=$$ sid=$(ps -o sid= -p $$ | tr -d ' ') base_lines=$BASE"

while true; do
  NEW=$(tail -n +$((BASE+1)) "$SUPPLY" 2>/dev/null | grep 'LAUNCH .*_qwen3_pd')
  BASE=$(wc -l < "$SUPPLY")
  if [ -n "$NEW" ]; then
    cell=$(echo "$NEW" | tail -1 | awk '{print $5}')
    log "NEW_LAUNCH: $NEW"
    sleep 420   # let it get past model load
    rl="outputs/$cell/run.log"
    if [ -f "$rl" ]; then
      echo "=== $cell run.log head ===" >> "$OUT"; head -40 "$rl" >> "$OUT"
      echo "=== $cell run.log tail ===" >> "$OUT"; tail -20 "$rl" >> "$OUT"
      if grep -Eq "(TypeError.*int|ValueError.*list|int' object|list index)" "$rl"; then
        log "int/list error signature found in $cell -- pkilling"
        pkill -f "outputs/$cell[ /]" && log "pkill sent for $cell"
      else
        log "no error signature in $cell"
      fi
    fi
  fi
  sleep 300
done
