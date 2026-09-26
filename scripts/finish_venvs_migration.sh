#!/bin/bash
# finish_venvs_migration.sh -- self-contained tail-end of the venvs migration
# /ultrafast_share/tsq/venvs -> /slow_share/tsq/venvs. Run under setsid nohup.
# Steps: wait for (or start) rsync -> verify (du + file count) -> swap via
# symlink -> smoke-test two venvs through the symlink -> log everything with
# timestamps to /slow_share/tsq/migrate_20260822.log.
set -u
SRC=/ultrafast_share/tsq/venvs
DST=/slow_share/tsq/venvs
LOG=/slow_share/tsq/migrate_20260822.log
SRC_PARENT=/ultrafast_share/tsq
STAMP=/ultrafast_share/tsq/venvs.migrated_20260822

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

du_bytes() { du -sB1 "$1" 2>/dev/null | awk '{print $1}'; }
nfiles()   { find "$1" -type f 2>/dev/null | wc -l; }

rsync_running() { pgrep -f 'rsync -a.*ultrafast_share/tsq/venvs/.*slow_share/tsq/venvs/' >/dev/null 2>&1; }

log "=== finish_venvs_migration start pid=$$ sid=$(ps -o sid= -p $$ | tr -d ' ') ==="

# ---- 1. wait for existing rsync; restart if none and sizes diverge >1% ----
while true; do
  if rsync_running; then
    log "rsync in progress, waiting (src_du=$(du_bytes "$SRC"))"
  else
    sd=$(du_bytes "$SRC") ; dd=$(du_bytes "$DST")
    if [ -z "$sd" ] || [ "$sd" = "0" ]; then log "FATAL: src du failed"; exit 1; fi
    diff_pct=$(( (sd - dd) * 100 / sd ))
    log "no rsync running: src=$sd dst=$dd diff=${diff_pct}%"
    if [ "$diff_pct" -le 1 ]; then break; fi
    log "starting our own rsync (diff>1%)"
    nohup rsync -a "$SRC/" "$DST/" >>"$LOG" 2>&1 </dev/null &
    log "rsync pid=$!"
  fi
  sleep 120
done

# ---- 2. verify ----
sd=$(du_bytes "$SRC") ; dd=$(du_bytes "$DST")
sf=$(nfiles "$SRC")   ; df=$(nfiles "$DST")
diff_pct=$(( (sd - dd) * 100 / sd ))
log "verify: du src=$sd dst=$dd diff=${diff_pct}% ; files src=$sf dst=$df (delta=$((sf-df)))"
if [ "$diff_pct" -gt 1 ] || [ "$((sf - df))" -gt 5 ] || [ "$((df - sf))" -gt 5 ]; then
  log "VERIFY FAILED -- leaving swap undone. See counts above."
  exit 2
fi
log "verify OK"

# ---- 3. swap (only if nothing is writing into src) ----
WRITERS=$(lsof +D "$SRC" 2>/dev/null | awk 'NR>1 && $4 ~ /[0-9]+w/ {print $1, $2}' | sort -u)
if [ -n "$WRITERS" ]; then
  log "writers detected in src, ABORTING swap: $WRITERS"
  exit 3
fi
if [ -e "$STAMP" ] || [ -L "$SRC" ]; then
  log "swap already done (stamp or symlink present), skipping"
else
  mv "$SRC" "$STAMP" && ln -s "$DST" "$SRC"
  rc=$?
  log "swap: mv->$STAMP && ln -s rc=$rc"
  [ $rc -ne 0 ] && exit 4
fi
ls -ld "$SRC" | tee -a "$LOG"

# ---- 4. smoke-test two venvs through the symlink ----
for v in $(ls "$STAMP" 2>/dev/null | head -2); do
  py="$SRC/$v/bin/python"
  if [ -x "$py" ]; then
    log "smoke $v transformers: $($py -c 'import transformers; print(transformers.__version__)' 2>&1 | tail -1)"
    log "smoke $v vllm: $($py -c 'import vllm; print(vllm.__version__)' 2>&1 | tail -1)"
  else
    log "smoke $v: no python at $py"
  fi
done

log "=== finish_venvs_migration done OK ==="
