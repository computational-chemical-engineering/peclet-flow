#!/bin/bash
# WO-W3 — TBFsolver's channel_18 on the block VoF container, ONE H100, in restartable chunks.
#
# Snellius conventions this follows (suite/docs/SNELLIUS.md): submit from THIS directory; every
# selector is a POSITIONAL argument (a leading `VAR=x sbatch` is silently dropped by SURF's
# sbatch); overrides that must be env (SUITE, BUILD, VENV) are `export`ed beforehand or passed
# with `sbatch --export=ALL,VAR=...`.
#
#   sbatch snellius_channel_18.sh <tag> [wall_seconds] [steps_cap] [turnover_target]
#
# The run is BILLED PER ALLOCATED GPU, and this case fits one: --gpus-per-node=1, never
# --exclusive.  Each chunk resumes from runs/<tag>/ckpt.npz and rewrites it; when the turnover
# target is reached the driver writes ckpt.npz.done and every later chunk in the chain exits
# immediately, so the chain can be over-provisioned safely.
#
#   jid=$(sbatch --parsable snellius_channel_18.sh prod 39000 200000 20)
#   for i in $(seq 1 5); do jid=$(sbatch --parsable --dependency=afterok:$jid \
#                                  snellius_channel_18.sh prod 39000 200000 20); done
#
#SBATCH --job-name=ch18w3
#SBATCH --account=tes24005
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=11:00:00
#SBATCH --output=ch18w3-%j.out

set -euo pipefail

TAG=${1:-prod}
WALLS=${2:-38000}        # seconds of stepping before the driver checkpoints and exits
STEPS=${3:-200000}       # per-chunk step cap (the wall budget normally binds first)
TURNS=${4:-20}           # eddy-turnover target for the WHOLE run

SUBMIT=${SLURM_SUBMIT_DIR:-$PWD}
SUITE=${SUITE:-/projects/0/prjs1022/peclet/suite}
FLOW=${FLOW:-$SUITE/flow-w3}
BUILD=${BUILD:-$FLOW/build_cuda}
VENV=${VENV:-$SUITE/flow/.venv}
RUNDIR=${RUNDIR:-$SUBMIT/runs/$TAG}
TBF=${TBF:-$SUITE/flow-w3/tests/study/channel_18/tbf_ic}

mkdir -p "$RUNDIR"
cd "$RUNDIR"

if [ -f ckpt.npz.done ]; then
  echo "[ch18] runs/$TAG already reached the turnover target (ckpt.npz.done) — nothing to do."
  exit 0
fi

module load 2024
module load gompi/2024a
module load CUDA/12.6.0
module load UCX-CUDA/1.16.0-GCCcore-13.3.0-CUDA-12.6.0
module load Python/3.12.3-GCCcore-13.3.0
export OMPI_MCA_pml=ucx UCX_MEMTYPE_CACHE=n
export OMP_NUM_THREADS=8 OMP_PROC_BIND=false        # the CUDA prefix carries an OpenMP HOST backend
export PYTHONPATH="$BUILD:${PYTHONPATH:-}"
export PECLET_FLOW_EXACT_RESIDUAL=1

echo "[ch18] $(date -Is)  job $SLURM_JOB_ID on $(hostname)  tag=$TAG"
echo "[ch18] build $BUILD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun --mpi=pmix --ntasks=1 --gpus-per-task=1 --gpu-bind=per_task:1 \
  "$VENV/bin/python" "$SUBMIT/run_channel_18.py" \
    --tbf "$TBF" --ny 80 \
    --ckpt "$RUNDIR/ckpt.npz" --out "$RUNDIR/stats.npz" \
    --chunklog "$RUNDIR/chunks.jsonl" \
    --steps "$STEPS" --wall "$WALLS" --turnovers "$TURNS" \
    --stats-start 4 --sample 10 --ckpt-every 2000

echo "[ch18] $(date -Is)  chunk finished"
ls -la "$RUNDIR"
