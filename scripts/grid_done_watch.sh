#!/bin/bash
# grid_done_watch.sh -- persistent setsid watchdog that records every DONE line
# from the grid supply daemon into outputs/grid_done_events.log with row counts,
# so cells finished during a disconnect can be registered later at a glance.
set -u
cd /home/tsq/rhq/omniconfess/OmniConfess
SUPPLY=outputs/grid_supply.log
EVENTS=outputs/grid_done_events.log
BASE=$(wc -l < "$SUPPLY" 2>/dev/null || echo 0)
echo "[done_watch $(date '+%F %T')] start pid=$$ base_lines=$BASE" >> "$EVENTS"

while true; do
  tail -n +$((BASE+1)) "$SUPPLY" 2>/dev/null | grep 'DONE ' | while read -r _ _ _ _ cell rest; do  # [supply] date time DONE cell ...
    scored="outputs/${cell}/${cell}_full_single_scored.jsonl"
    rows=$(wc -l < "$scored" 2>/dev/null || echo 0)
    echo "$(date '+%F %T') cell=$cell $rest scored_rows=$rows" >> "$EVENTS"
  done
  BASE=$(wc -l < "$SUPPLY")
  sleep 120
done
