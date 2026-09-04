#!/bin/bash
# WO-W3 build validation on ONE H100 (short): the channel_18 driver's --quick mode, plus the gate
# the W3 machinery actually needs — a CHUNKED run must equal the continuous one.
#
#   A: 40 steps in one go                      -> ckptA.npz
#   B: 20 steps, then RESUME for 20 more       -> ckptB.npz
#   compare A and B cell by cell (velocity and every marker's own colour)
#
#   sbatch snellius_validate.sh
#
#SBATCH --job-name=ch18val
#SBATCH --account=tes24005
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=ch18val-%j.out

set -euo pipefail

SUBMIT=${SLURM_SUBMIT_DIR:-$PWD}
SUITE=${SUITE:-/projects/0/prjs1022/peclet/suite}
FLOW=${FLOW:-$SUITE/flow-w3}
BUILD=${BUILD:-$FLOW/build_cuda}
VENV=${VENV:-$SUITE/flow/.venv}
RUNDIR=${RUNDIR:-$SUBMIT/runs/validate}
TBF=${TBF:-$FLOW/tests/study/channel_18/tbf_ic}

module load 2024
module load gompi/2024a
module load CUDA/12.6.0
module load UCX-CUDA/1.16.0-GCCcore-13.3.0-CUDA-12.6.0
module load Python/3.12.3-GCCcore-13.3.0
export OMPI_MCA_pml=ucx UCX_MEMTYPE_CACHE=n
export OMP_NUM_THREADS=8 OMP_PROC_BIND=false
export PYTHONPATH="$BUILD:${PYTHONPATH:-}"

rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"; cd "$RUNDIR"
PY="$VENV/bin/python"
DRV="$SUBMIT/run_channel_18.py"
COMMON="--quick --ny 48 --tbf $TBF --sample 4 --ckpt-every 1000"

echo "=== A: 40 steps in one go ==="
srun --ntasks=1 --gpus-per-task=1 "$PY" "$DRV" $COMMON --steps 40 \
     --ckpt ckptA.npz --out statsA.npz --chunklog chunksA.jsonl

echo "=== B1: 20 steps ==="
srun --ntasks=1 --gpus-per-task=1 "$PY" "$DRV" $COMMON --steps 20 \
     --ckpt ckptB.npz --out statsB.npz --chunklog chunksB.jsonl
echo "=== B2: resume for 20 more ==="
srun --ntasks=1 --gpus-per-task=1 "$PY" "$DRV" $COMMON --steps 20 \
     --ckpt ckptB.npz --out statsB.npz --chunklog chunksB.jsonl

echo "=== restart gate: A vs B ==="
"$PY" - <<'EOF'
import numpy as np
a, b = np.load("ckptA.npz"), np.load("ckptB.npz")
assert int(a["step"]) == int(b["step"]) == 40, (int(a["step"]), int(b["step"]))
worst = 0.0
for k in ("u", "v", "w"):
    d = float(np.max(np.abs(a[k] - b[k])))
    s = float(np.max(np.abs(a[k]))) or 1.0
    worst = max(worst, d / s)
    print(f"  {k}: max|dA-B| {d:.3e}  (rel {d/s:.3e})")
print(f"  t: {float(a['t']):.9e} vs {float(b['t']):.9e}")
n = int(a["nblocks"])
assert n == int(b["nblocks"])
dc = 0.0
for i in range(n):
    assert list(a[f"box{i}"]) == list(b[f"box{i}"]), (i, a[f"box{i}"], b[f"box{i}"])
    dc = max(dc, float(np.max(np.abs(a[f"col{i}"] - b[f"col{i}"]))))
print(f"  {n} markers: identical boxes, max|dC| {dc:.3e}")
ok = worst < 1e-12 and dc < 1e-12
print("RESTART GATE:", "PASS (bitwise or at the float floor)" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
EOF
echo "[validate] done $(date -Is)"
