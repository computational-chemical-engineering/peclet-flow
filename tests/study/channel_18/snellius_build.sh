#!/bin/bash
# WO-W3 — build the CUDA (HOPPER90) flow module for the channel_18 run, in this session's OWN
# directory ($SUITE/flow-w3 against $SUITE/core-w3): the shared $SUITE/flow tree is a collision
# zone with other sessions and is never touched.  No GPU is needed to compile, so this runs on the
# CPU partition.
#
#   sbatch snellius_build.sh          # -> $FLOW/build_cuda/peclet/flow/_flow*.so
#
#SBATCH --job-name=ch18build
#SBATCH --account=tes24005
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:30:00
#SBATCH --output=ch18build-%j.out

set -euo pipefail

SUITE=${SUITE:-/projects/0/prjs1022/peclet/suite}
FLOW=${FLOW:-$SUITE/flow-w3}
CORE=${CORE:-$SUITE/core-w3}
VENV=${VENV:-$SUITE/flow/.venv}
BUILD=${BUILD:-$FLOW/build_cuda}
FRESH=${FRESH:-1}

module load 2024
module load gompi/2024a
module load CUDA/12.6.0
module load UCX-CUDA/1.16.0-GCCcore-13.3.0-CUDA-12.6.0
module load Python/3.12.3-GCCcore-13.3.0
export OMPI_MCA_pml=ucx UCX_MEMTYPE_CACHE=n

PY="$VENV/bin/python"
INC=$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY"))')

[ "$FRESH" = "1" ] && rm -rf "$BUILD"      # FindPython's artifact variables are sticky

cmake -S "$FLOW" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$SUITE/extern/install/nvidia-cuda" \
  -DPECLET_SIBLING_PECLET_CORE="$CORE" \
  -DTPX_DIR="$CORE" \
  -DPython_EXECUTABLE="$PY" \
  -DPython_INCLUDE_DIR="$INC"

cmake --build "$BUILD" -j "${SLURM_CPUS_PER_TASK:-16}"
ls -la "$BUILD"/peclet/flow/
echo "[build] done $(date -Is)"
