#!/bin/bash
# Run any command in the kalmamba image on one GPU, from the repository root:
#   sbatch scripts/isambard/run.sbatch python scripts/bench_runtime.py
#SBATCH --job-name=kalmamba-run
#SBATCH --output=kalmamba-run-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
set -euo pipefail
podman-hpc run --rm --gpu -e PYTHONPATH=/opt/kalmamba -e TRITON_CACHE_DIR="/tmp/triton-${SLURM_JOB_ID}" \
    -v "${PWD}:/opt/kalmamba" -w /opt/kalmamba "${IMAGE:-kalmamba:v1}" "$@"
