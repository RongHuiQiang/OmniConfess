#!/bin/bash
# run_claim_helper.sh -- shared warm-claim launcher for 30B single-card runs
# (source this). Usage after sourcing:
#   claim_run <gpu> <venv_py> <warm_gb> <logfile> <workdir> -- <cmd...>
# Pattern copied from /ultrafast_share/tsq/tsq/nemotron/chain_nemotron_v2.sh:
# a 55GB holder covers the python-import phase; the run pre-claims <warm_gb>
# via warm_claim.py at process start so the card never looks free during the
# multi-minute .to() ramp. Callers whose runner empty_cache()s per item must
# export NEMO_KEEP_CACHE=1 (that env is what keeps the reservation).
WARMDIR=/ultrafast_share/tsq/tsq/nemotron
CLAIMDIR=${CLAIMDIR:-/tmp/xbackbone_claims}
mkdir -p "$CLAIMDIR"

claim_run() {
  local gpu=$1 py=$2 warm_gb=$3 log=$4 wd=$5; shift 5
  [ "$1" = "--" ] && shift
  # absolute script path: runpy.run_path sets sys.path[0] to the script dir,
  # which a relative path leaves unresolvable for the runner's `import shared`
  case "$1" in
    /*) ;;
    *) set -- "$wd/$1" "${@:2}" ;;
  esac
  local ready=$CLAIMDIR/.ready_gpu${gpu}_$$
  local gone=$CLAIMDIR/.gone_gpu${gpu}_$$
  rm -f "$ready" "$gone"
  "$py" "$WARMDIR/hold_gpu.py" "$gpu" 55 >"$CLAIMDIR/holder_gpu${gpu}_$$.log" 2>&1 &
  local hpid=$!
  for i in $(seq 1 30); do
    grep -q '\[holder\]' "$CLAIMDIR/holder_gpu${gpu}_$$.log" 2>/dev/null && break
    sleep 1
  done
  ( cd "$wd" && CUDA_VISIBLE_DEVICES=$gpu "$py" "$WARMDIR/warm_claim.py" \
      "$warm_gb" "$ready" "$gone" "$@" ) >"$log" 2>&1 &
  local rpid=$!
  for i in $(seq 1 300); do
    kill -0 $rpid 2>/dev/null || break
    [ -f "$ready" ] && break
    sleep 1
  done
  kill $hpid 2>/dev/null
  touch "$gone"
  for i in $(seq 1 90); do
    grep -q '\[warm\]' "$log" 2>/dev/null && break
    kill -0 $rpid 2>/dev/null || break
    sleep 1
  done
  if grep -q '\[warm\]' "$log" 2>/dev/null; then
    echo "[claim $(date +%T)] gpu $gpu: ${warm_gb}GB warm claim confirmed"
  else
    echo "[claim $(date +%T)] gpu $gpu: WARN warm claim NOT confirmed"
  fi
  wait $rpid
  return $?
}
