#!/bin/bash
# WO-W3 — pull the Snellius results of the channel_18 campaign back to the workstation and plot.
#
#   bash collect.sh [local target dir]           # default: ./results
#
# What it fetches:
#   peclet     $SUITE/flow-w3/tests/study/channel_18/runs/prod/{stats.npz,chunks.jsonl}
#              + the chunk logs ch18w3-<jobid>.out   (ckpt.npz is 16 MB and stays remote)
#   TBFsolver  /projects/0/prjs1022/peclet/TBFsolver/run_prod/<last folder>/stats_*  + tauw
#              (converted to the same NPZ keys by tbf_profiles.py)
set -eu

DST=${1:-results}
HOST=${HOST:-snellius}
SUITE=${SUITE:-/projects/0/prjs1022/peclet/suite}
CH18=$SUITE/flow-w3/tests/study/channel_18
TBF=${TBF:-/projects/0/prjs1022/peclet/TBFsolver/run_prod}

mkdir -p "$DST"
echo "== peclet =="
rsync -av "$HOST:$CH18/runs/prod/stats.npz" "$HOST:$CH18/runs/prod/chunks.jsonl" "$DST/" || true
rsync -av --include='ch18w3-*.out' --exclude='*' "$HOST:$CH18/" "$DST/logs/" || true

echo "== TBFsolver: the LAST written output folder =="
LAST=$(ssh "$HOST" "cd $TBF 2>/dev/null && ls -d [0-9]* 2>/dev/null | sort -n | tail -1 | sed 's|^|$TBF/|'")
echo "  $LAST"
if [ -n "$LAST" ]; then
  mkdir -p "$DST/tbf"
  rsync -av --include='stats_*' --include='tauw' --exclude='*' "$HOST:$LAST/" "$DST/tbf/" || true
  python3 "$(dirname "$0")/tbf_profiles.py" "$DST/tbf" --out "$DST/tbf_profiles.npz" || true
fi

echo "== plot =="
if [ -f "$DST/stats.npz" ]; then
  ARGS=("$DST/stats.npz" --out "$DST/channel_18.png")
  [ -f "$DST/tbf_profiles.npz" ] && ARGS+=(--tbf "$DST/tbf_profiles.npz")
  python3 "$(dirname "$0")/plot_channel_18.py" "${ARGS[@]}"
fi
echo "done -> $DST"
