# Running KalMamba on Isambard-AI

Isambard-AI has no Docker, so the image is built and run with `podman-hpc`.
The code is mounted into the container at run time, so code changes need no rebuild.

## Build the image

From the repository root:

```bash
sbatch --wait scripts/isambard/build_image.sbatch
```

The job takes about 11 minutes, most of it compiling `causal-conv1d` and `mamba-ssm`.
Its log is `kalmamba-build-<jobid>.out`, and a working image ends with `SMOKE TEST PASSED`.

## Run a command

Set `PYTHONPATH` so that scripts outside the repository root can import `ssm_rl`:

```bash
srun --nodes=1 --gpus=1 --time=20 podman-hpc run --rm --gpu -e PYTHONPATH=/opt/kalmamba -v "$PWD:/opt/kalmamba" -w /opt/kalmamba kalmamba:v1 python scripts/bench_kalman_scan.py
```

Run the tests:

```bash
srun --nodes=1 --gpus=1 --time=10 podman-hpc run --rm --gpu -v "$PWD:/opt/kalmamba" -w /opt/kalmamba kalmamba:v1 python -m pytest -q -p no:cacheprovider tests
```

## Version notes

- `causal-conv1d` and `mamba-ssm` are built from their GitHub tags because the PyPI source archives lack the C++ sources.
- `transformers` is pinned to 4.46.3 because `mamba-ssm` 2.2.4 imports a class that later releases removed.
- `dm_control` 1.0.28 is paired with `mujoco` 3.3.0, the version its `requirements.txt` pins. Other pairs fail with missing `MjModel` fields.
- `apt-get update` has timeouts because a stalled Ubuntu mirror once hung a build for an hour.
