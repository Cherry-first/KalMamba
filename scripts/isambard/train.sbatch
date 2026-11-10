#!/bin/bash
# Run one experiment from a cw2 config in the kalmamba image.
# Run from the repository root:
#   sbatch scripts/isambard/train.sbatch experiments/configs/kalmamba/noisy_state.yml kalmamba
# Results go to the config's `path`, under the experiment name.
#SBATCH --job-name=kalmamba-train
#SBATCH --output=kalmamba-train-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00

set -euo pipefail

CONFIG="$1"
EXPERIMENT="$2"
IMAGE="${IMAGE:-kalmamba:v1}"

# Each job gets its own Triton cache: jobs that started together on one node crashed with
# "PyInit___triton_launcher" when they shared one.
podman-hpc run --rm --gpu -e PYTHONPATH=/opt/kalmamba -e TRITON_CACHE_DIR="/tmp/triton-${SLURM_JOB_ID}" \
    -e KALMAMBA_DIAG_EVERY="${KALMAMBA_DIAG_EVERY:-0}" \
    -v "${PWD}:/opt/kalmamba" -w /opt/kalmamba "${IMAGE}" \
    python experiments/run_rl.py "${CONFIG}" -e "${EXPERIMENT}" -o

# cw2 catches exceptions and exits 0, so fail the job here if the run raised one.
if grep -q "^Traceback" "kalmamba-train-${SLURM_JOB_ID}.out"; then
    echo "The run raised an exception, see the traceback above." >&2
    exit 1
fi
