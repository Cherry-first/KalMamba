#!/bin/bash
# Build, migrate and smoke test the KalMamba image on a compute node.
# Run from the repository root:  sbatch --wait scripts/isambard/build_image.sbatch
#
# This runs on a compute node because login nodes cap each user at 500 tasks,
# which makes the squash step of `podman-hpc migrate` fail with
# "FATAL ERROR: Failed to create thread".
#SBATCH --job-name=kalmamba-build
#SBATCH --output=kalmamba-build-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=120

set -euo pipefail

IMAGE="${IMAGE:-kalmamba:v1}"

# Rebuilding a tag leaves two squashed images with one name, which breaks the store.
podman-hpc rmsqi "${IMAGE}" || true
podman-hpc build . --tag "${IMAGE}"
podman-hpc migrate "${IMAGE}"
podman-hpc images
podman-hpc run --rm --gpu -v "${PWD}:/opt/kalmamba" -w /opt/kalmamba "${IMAGE}" python scripts/isambard/smoke_test.py
