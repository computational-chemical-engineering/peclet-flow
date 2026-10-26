#!/bin/bash
# run_matrix.sh <module_dir_cuda> <module_dir_host> <out_dir> [rows...]
#
# The WO-0 measurement matrix of doc/mg_depth_policy.md §6 (rerun by the WO-2 gates), blind: every
# run logs to <out_dir>/<row>/, and the script prints one summary line per (row, depth) plus the
# M1 checkpoint (G-PERF-WS (a)-(c)) when M1 ran. Rows (default: all):
#   M1  swarm 192^3 device, MG-PCG rtol 1e-8          L 4 5 6 7
#   M2  bubble column 256x192x128 device (fresh start) L 4 5 6
#   M3z zh_sphere N=128 device                         L 4 5 6
#   M3r random_spheres N=128 device                    L 4 5 6
#   M3s zh_sphere + random_spheres N=48 device, the solver default depth (L 0), with dumps
#   M4  channel 2048x64x64 device, single rank         L 4 5 6
#   M5  swarm 128^3 HOST module, 1 rank x 6 threads    L 4 5 6
#   M6  anchored 128x32x32 inflow/outflow device       L 4 5 6
#   M7a swarm 192^3 device, the DEFAULT driver         L 4 5 6
#   M7b porous bed 128^3 device, CFD-DEM style (drag), the default driver   L 4 5 6
#   M7c porous bed 128^3 device without drag, the default driver            L 4 5 6
#   M8  channel 2048x64x64 device, np 2 (standalone V-cycle)                L 4 5 6
#   M9z zh_sphere N=128 device, SolverColocated AUTO ('ghost': BiCGStab)    L 4 5
#   M9r random_spheres N=128 device, SolverColocated AUTO ('ghost')         L 4 5
#   M10  swarm 128x128x25 device (z stops coarsening at once), MG-PCG       L 0 4 6
#   M10h swarm 128x128x25 HOST module, 1 rank x 6 threads, the default driver (Chebyshev)  L 0 4 6
#   M10v channel 512x64x25 HOST module, np 2 x 3 threads (z between walls)  L 0 4 6
#   M10b swarm 512x512x100 device (z stops at 25)                           L 0 4
# Environment (runner settings only; none reaches the solver):
#   EXTRA_LEVELS="0"  depths prepended to every row's list (0 = the solver default: the WO-2 column)
#   ROUNDS=3          interleaved A/B rounds per row (median over rounds)
#   DUMP=1            also --dump every run (large at 192^3); M3s always dumps
#   LR_OFF_LEVELS="0 4"  extra arms at these depths with --line-relaxation off (doc/mg_depth_policy.md
#                     A2.5 G-LR), run and labelled as depth tokens "<L>off" (files *_L<L>off_*)
#   LOCK=<file>       the heavy-job lock (default bubble_column_perf/gpu_diag/mgd/heavy.lock)
# Each row = one probe per depth (2 steps, PECLET_FLOW_MG_DEBUG=1 PECLET_FLOW_AGMG_DEBUG=1: debug
# output only, read for the bottom engine) and ROUNDS timed rounds L_a L_b ... L_a L_b ..., all
# under ONE hold of the lock.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
LOCK=${LOCK:-/home/frankp/Codes/bubble_column_perf/gpu_diag/mgd/heavy.lock}
ROUNDS=${ROUNDS:-3}
EXTRA_LEVELS=${EXTRA_LEVELS:-}
LR_OFF_LEVELS=${LR_OFF_LEVELS:-}
DUMP=${DUMP:-0}

if [ "${1:-}" = "__row" ]; then   # inner: one row, called under flock
  shift; MC=$1; MH=$2; OUT=$3; ROW=$4
  source /home/frankp/Codes/suite/.venv/bin/activate
  D=$OUT/$ROW; mkdir -p "$D"
  thr=4; mod=$MC; launch="python"; steps=30; warm=5
  case $ROW in
    M1)  args="--case swarm --cells 192"; lv="4 5 6 7" ;;
    M2)  args="--case bc --cells-per-d 32"; lv="4 5 6"; warm=10 ;;
    M3z) args="--case zh_sphere --n 128"; lv="4 5 6" ;;
    M3r) args="--case random_spheres --n 128"; lv="4 5 6" ;;
    M3s) args=""; lv="0" ;;
    M9z) args="--case zh_sphere --n 128 --solver colocated"; lv="4 5" ;;
    M9r) args="--case random_spheres --n 128 --solver colocated"; lv="4 5" ;;
    M10) args="--case swarm --dims 128 128 25"; lv="0 4 6" ;;
    M10h) args="--case swarm --dims 128 128 25 --driver default"; lv="0 4 6"; mod=$MH; thr=6 ;;
    M10v) args="--case channel --nx 512 --ny 64 --nz 25"; lv="0 4 6"; mod=$MH; thr=3
          launch="mpirun -np 2 --bind-to none -x OMP_NUM_THREADS -x OMP_PROC_BIND -x PYTHONPATH python" ;;
    M10b) args="--case swarm --dims 512 512 100"; lv="0 4" ;;
    M4)  args="--case channel"; lv="4 5 6" ;;
    M5)  args="--case swarm --cells 128"; lv="4 5 6"; mod=$MH; thr=6 ;;
    M6)  args="--case anchored"; lv="4 5 6" ;;
    M7a) args="--case swarm --cells 192 --driver default"; lv="4 5 6" ;;
    M7b) args="--case porous --n 128"; lv="4 5 6" ;;
    M7c) args="--case porous --n 128 --no-drag"; lv="4 5 6" ;;
    M8)  args="--case channel"; lv="4 5 6"; thr=2
         launch="mpirun -np 2 --bind-to none -x OMP_NUM_THREADS -x OMP_PROC_BIND -x PYTHONPATH python" ;;
    *) echo "ROW $ROW unknown"; exit 2 ;;
  esac
  [ -n "$EXTRA_LEVELS" ] && lv="$EXTRA_LEVELS $lv"
  for L in $LR_OFF_LEVELS; do lv="$lv ${L}off"; done
  export OMP_NUM_THREADS=$thr OMP_PROC_BIND=false PYTHONPATH=$mod
  export PATH=/usr/local/cuda-13.4/bin:$PATH
  subcases=("$args")
  [ "$ROW" = M3s ] && subcases=("--case zh_sphere --n 48" "--case random_spheres --n 48")
  echo "$ROW module=$mod threads=$thr levels=[$lv] rounds=$ROUNDS start $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)" > "$D/session.txt"
  si=0
  for sa in "${subcases[@]}"; do
    sub=$ROW; [ ${#subcases[@]} -gt 1 ] && sub=${ROW}_$si
    for L in $lv; do   # probes: the bottom engine from the debug lines
      PECLET_FLOW_MG_DEBUG=1 PECLET_FLOW_AGMG_DEBUG=1 $launch "$HERE/mgd_case.py" $sa --levels ${L%off} $([[ $L == *off ]] && echo --line-relaxation off) \
        --steps 2 --warm 0 --max-steps 2 --tag probe > "$D/${sub}_L${L}_probe.log" 2>&1
      echo "probe rc $?" >> "$D/${sub}_L${L}_probe.log"
    done
    for r in $(seq 1 "$ROUNDS"); do
      for L in $lv; do
        ex=""; { [ "$DUMP" = 1 ] || [ "$ROW" = M3s ]; } && [ "$r" = 1 ] && ex="--dump $D/${sub}_L${L}.npz"
        gb=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        $launch "$HERE/mgd_case.py" $sa --levels ${L%off} $([[ $L == *off ]] && echo --line-relaxation off) --steps $steps --warm $warm --tag "${sub}_r$r" \
          --json "$D/${sub}_L${L}_r$r.json" $ex > "$D/${sub}_L${L}_r$r.log" 2>&1
        echo "rc $? gpu_apps_before $gb gpu_apps_after $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l) load_after $(cut -d" " -f1 /proc/loadavg)" >> "$D/${sub}_L${L}_r$r.log"
      done
    done
    si=$((si + 1))
  done
  echo "end $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$D/session.txt"
  exit 0
fi

if [ $# -lt 3 ]; then
  sed -n 2,26p "$0"; exit 2
fi
MC=$(cd "$1" && pwd); MH=$(cd "$2" && pwd); OUT=$3; shift 3
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
ROWS=${*:-M1 M2 M3z M3r M3s M4 M5 M6 M7a M7b M7c M8}
for ROW in $ROWS; do
  flock "$LOCK" bash "$0" __row "$MC" "$MH" "$OUT" "$ROW"
  source /home/frankp/Codes/suite/.venv/bin/activate
  python "$HERE/summarize.py" "$OUT" "$ROW"
done
echo "MATRIX_EXIT $(date -Is)"
