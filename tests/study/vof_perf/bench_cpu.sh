#!/bin/bash
# CPU timings of the bubble column from ckpt_t43, MG-PCG pressure (set_pressure_pcg(True, 800, 1e-10))
# -- the VoF step-performance harness (WO-0, doc/vof_step_performance_design.md sec. 8 G-PERF).
# RUN ON A QUIET HOST (load average < ~2, or a Snellius node): numbers taken at load average 50-100
# (2026-09-25, other sessions' jobs) are indicative only.
#   ROUNDS=2 bash tests/study/vof_perf/bench_cpu.sh > bench_cpu.log 2>&1   (~20 min per round, quiet)
#   TREE_A=<dir> LABEL_A=... TREE_B=<dir> ... CKPT=<npz> bash tests/study/vof_perf/bench_cpu.sh
# Up to three host-openmp PECLET_FLOW_MPI=ON trees (a tree left empty is skipped); each is a
# PYTHONPATH that imports peclet.flow. Defaults: A = the frozen WO-0 baseline (flow ed05b6f),
# B = the vof-mg worktree's build_omp, C = none. (The 2026-09-25 comparison against flow b0569d6
# needed a pre-set_bulk_velocity copy of the case driver, run_peclet_old.py; that branch is gone.)
# Output: one "ms/step" line per configuration + the per-stage breakdown of the LAST tree given.
set -u
P=$(cd "$(dirname "$0")" && pwd)
SUITE=/home/frankp/Codes/suite
TREE_A=${TREE_A-/home/frankp/Codes/bubble_column_perf/frozen_base/omp}
LABEL_A=${LABEL_A-"WO-0 baseline (ed05b6f)"}
TREE_B=${TREE_B-$SUITE/flow-vof-mg/build_omp}
LABEL_B=${LABEL_B-"vof-mg build_omp"}
TREE_C=${TREE_C-}
LABEL_C=${LABEL_C-"C"}
CKPT=${CKPT-$HOME/Codes/bubble_column_perf/ckpt_t43.npz}
source "$SUITE/.venv/bin/activate"
cd "$P"
echo "load: $(cat /proc/loadavg)"
run1() {  # label, pythonpath[, threads = 24]
  echo "== 1 rank x ${3:-24} threads: $1"
  OMP_NUM_THREADS=${3:-24} OMP_PROC_BIND=spread OMP_PLACES=cores PYTHONPATH="$2" \
    python prof.py 20 --warm 3 --pcg --flux device --ckpt "$CKPT" 2>&1 | grep -E "ms/step|iters"
}
run8() {  # label, pythonpath
  echo "== 8 ranks x 3 threads (TBFsolver's layout): $1"
  OMP_NUM_THREADS=3 OMP_PROC_BIND=true OMP_PLACES=cores PYTHONPATH="$2" \
    mpirun -np 8 --map-by ppr:8:node:pe=3 --bind-to core -x OMP_NUM_THREADS -x OMP_PROC_BIND \
    -x OMP_PLACES -x PYTHONPATH python run_mpi.py 20 --warm 3 --ckpt "$CKPT" 2>&1 | grep -E "ms/step"
}
LAST=
# ROUNDS interleaved rounds (A B C A B C ...) so a drifting background load biases no one tree;
# quote the per-configuration minimum.
for r in $(seq 1 "${ROUNDS:-2}"); do
  echo "#### round $r  load: $(cat /proc/loadavg)"
  for threads in 24 8; do
    [ -n "$TREE_A" ] && run1 "$LABEL_A" "$TREE_A" $threads
    [ -n "$TREE_B" ] && run1 "$LABEL_B" "$TREE_B" $threads
    [ -n "$TREE_C" ] && run1 "$LABEL_C" "$TREE_C" $threads
  done
  [ -n "$TREE_A" ] && { run8 "$LABEL_A" "$TREE_A"; LAST=$TREE_A; }
  [ -n "$TREE_B" ] && { run8 "$LABEL_B" "$TREE_B"; LAST=$TREE_B; }
  [ -n "$TREE_C" ] && { run8 "$LABEL_C" "$TREE_C"; LAST=$TREE_C; }
done
echo "== $LAST, 1 x 24, per-stage (set_vof_timing)"
OMP_NUM_THREADS=24 OMP_PROC_BIND=spread OMP_PLACES=cores PYTHONPATH="$LAST" \
  python prof.py 20 --warm 3 --pcg --flux device --timing --ckpt "$CKPT" 2>&1 | sed -n '/ms\/step/,$p'
echo "load: $(cat /proc/loadavg)"
