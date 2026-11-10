# KalMamba reproduction

An independent reproduction of

> Philipp Becker, Niklas Freymuth, Gerhard Neumann.
> **KalMamba: Towards Efficient Probabilistic State Space Models for RL under Uncertainty.**
> arXiv:2406.15131, 2024.

The paper has no official code. This repository builds KalMamba on top of
[CoRAL](https://github.com/pbecker93/CoRAL), the first author's earlier codebase, whose default
configs match the paper's hyperparameter tables. The VRKN baseline is ported from
[pbecker93/vrkn](https://github.com/pbecker93/vrkn). The original CoRAL README is in
[README_CORAL.md](README_CORAL.md).

`docs/kalmamba_explainer.html` is an introduction to the model, in Chinese, for readers new to state
space models.

## Status

| Paper result | Status |
|---|---|
| Fig. 6: runtime against sequence length | Reproduced |
| Fig. 5: wall-clock training at sequence lengths 32 to 256 | Runtime part reproduced; performance part partly reproduced |
| Fig. 4 / 9: noisy state-based tasks | Partly reproduced: SAC fails, but KalMamba only ties RSSM and trails VRKN |
| Fig. 3 / 7 / 8: image tasks and ablations | Seed 0 running |

Details are under [Results](#results).

## What is in the repository

| Path | Contents |
|---|---|
| `ssm_rl/kalmamba/parallel_kalman.py` | Kalman filter and RTS smoother as parallel associative scans |
| `ssm_rl/kalmamba/transition.py` | Mamba backbone and the dynamics head that outputs A, b and Σ for each step |
| `ssm_rl/kalmamba/objective.py` | Smoothing ELBO (reconstruction, reward, balanced KL with free nats) and the Mahalanobis regularizer (Eq. 2) |
| `ssm_rl/kalmamba/vrkn.py` | VRKN baseline: sequential filter, linearized around the filtered mean |
| `ssm_rl/kalmamba/kalmamba_factory.py` | Builds the models and objective from a config |
| `ssm_rl/kalmamba/diagnostics.py` | Optional check that compares the beliefs seen during training and during acting |
| `experiments/configs/kalmamba/` | Experiment configs, one per figure; see below |
| `scripts/` | Sweep generation, runtime benchmark, aggregation and plotting |
| `scripts/isambard/` | Container build and Slurm job scripts for Isambard-AI |
| `tests/` | Tests for the parallel scans, against a sequential reference, and for the model |
| `experiments/outputs/` | Result tables, figures and Slurm job lists. Run logs are not committed. |

Changes to the original CoRAL files:
- `experiments/run_rl.py`: registers the `kalmamba` and `vrkn` world models.
- `experiments/env_factory.py`: adds observation noise and transition noise (Section 5.2).
- `experiments/rl_experiment.py`: logs wall-clock time and runs the optional diagnostic.
- `abstract_trainer.py` and `sac_trainer.py`: log the world-model gradient norm.
- `envs/make_env.py`: imports ManiSkill only when it is used.

## Setup

The code runs in a container. `Dockerfile` targets linux/arm64 with NVIDIA GH200, the hardware of
Isambard-AI. It installs:
- CUDA 12.8 and PyTorch 2.7.1;
- `mamba-ssm` 2.2.4 and `causal-conv1d` 1.5.0, compiled from source for sm_90;
- `dm_control` 1.0.28 with `mujoco` 3.3.0;
- `requirements.txt`.

On another GPU, change `TORCH_CUDA_ARCH_LIST` in the Dockerfile.

On Isambard-AI (podman-hpc), from the repository root:

```bash
sbatch --wait scripts/isambard/build_image.sbatch
```

The build takes about 11 minutes and ends with `SMOKE TEST PASSED`. The code is mounted into the
container at run time, so code changes need no rebuild. See
[scripts/isambard/README.md](scripts/isambard/README.md) for running commands and tests and for the
version pins.

Run the tests:

```bash
sbatch scripts/isambard/run.sbatch python -m pytest -q -p no:cacheprovider tests
```

## Running experiments

Each run is a [cw2](https://github.com/ALRhub/cw2) experiment: a config file plus an experiment name.
One run uses one GPU:

```bash
sbatch scripts/isambard/train.sbatch experiments/configs/kalmamba/noisy_state.yml kalmamba
```

Results go to the config's `path`. The Slurm log `kalmamba-train-<jobid>.out` holds the evaluation
returns. Without Slurm, run the command inside the container directly:

```bash
python experiments/run_rl.py <config> -e <experiment> -o
```

| Figure | Config | Experiments |
|---|---|---|
| Fig. 4 / 9: noisy state tasks | `noisy_state.yml`, `noisy_state_sac.yml` | `kalmamba`, `kalmamba_mean`, `kalmamba_noreg`, `vrkn`, `rssm`, `sac` |
| Fig. 5: sequence length | `fig5_seq_len.yml` | `{kalmamba,vrkn,rssm}_L{32,64,128,256}` |
| Fig. 3 / 7 / 8: image tasks and ablations | `image.yml` | `kalmamba`, `kalmamba_mean`, `no_mamba`, `no_mcd`, `no_reg`, `vrkn`, `rssm` |
| Studies of our own | `regularizer_study.yml`, `train_act_diag.yml` | see the comments in each file |

Every config also has `smoke_*` experiments, which run 3 iterations as an end-to-end check.

The per-task and per-seed sweeps (`noisy_state_sweep.yml`, `noisy_state_sac_sweep.yml`,
`image_sweep.yml`) are generated from these configs. Experiment names are `<method>_<task>_s<seed>`,
for the tasks cartpole-swingup, quadruped-walk, walker-walk and walker-run, and seeds 0 to 3:

```bash
python scripts/make_sweep_configs.py
```

```bash
sbatch scripts/isambard/train.sbatch experiments/configs/kalmamba/noisy_state_sweep.yml kalmamba_noreg_walker-run_s0
```

A full run is 1M environment steps (1001 iterations) and takes about 5 h for KalMamba on states, 9 to
10 h for KalMamba on images, and up to 15 h for VRKN on images. VRKN and RSSM need `--time=24:00:00`,
the cluster's limit, at longer sequence lengths.

### Analysis

The aggregation and plotting scripts read job lists of the form `<name> <slurm job id>`, such as
`experiments/outputs/noisy_state_jobs.txt`, together with the Slurm logs.

Fig. 4 and 9: mean over seeds with 95% stratified bootstrap confidence intervals.

```bash
python scripts/aggregate_results.py --out experiments/outputs/noisy_state_results
```

Fig. 5:

```bash
python scripts/plot_fig5.py
```

Fig. 6: benchmark on a GPU, then plot.

```bash
sbatch scripts/isambard/run.sbatch python scripts/bench_runtime.py
```

```bash
python scripts/plot_runtime.py experiments/outputs/runtime/fig6.csv experiments/outputs/runtime/fig6.png
```

## Results

### Fig. 6: runtime

Batch size 32 on one GH200. Times are in seconds at sequence length 2048.

| | KalMamba | RSSM | VRKN |
|---|---|---|---|
| Forward pass | 0.028 | 1.25 | 1.37 |
| Full training step (model and SAC) | 0.146 | 5.65 | 8.11 |

As in the paper, the baselines grow linearly with sequence length and KalMamba almost
logarithmically. Data and plot: `experiments/outputs/runtime/`.

### Fig. 4 / 9: noisy state-based tasks

Settings:
- DMC tasks with Gaussian noise of std 0.3 added to the observations and to the actions;
- 1M environment steps, 4 seeds;
- final return is the mean of the last 3 evaluations, with 95% stratified bootstrap CIs.

| Method | cartpole-swingup | quadruped-walk | walker-walk | walker-run | Aggregate |
|---|---|---|---|---|---|
| KalMamba (no regularizer) | 530 | 830 | 898 | 432 | 673 [633, 699] |
| RSSM + SAC | 498 | 775 | 956 | 500 | 682 [608, 729] |
| VRKN + SAC | 691 | 930 | 881 | 560 | 765 [724, 792] |
| SAC | 243 | 70 | 511 | 194 | 254 [235, 269] |

Per-task CIs: `experiments/outputs/noisy_state_results/final_returns.md`.

Compared with the paper:
- Plain SAC fails under noise, as in the paper.
- The paper says KalMamba clearly beats RSSM and almost matches VRKN. Here KalMamba ties RSSM (the
  CIs overlap) and trails VRKN on 3 of 4 tasks.

The KalMamba row is **not** the paper's configuration. The paper uses the Mahalanobis regularizer with
α = 1. Taken literally, with Eq. 2 summed over the 230 latent dimensions, the regularizer grows to
about 25,000 and dominates the loss. Its gradient norm reaches about 10⁴ against a clip norm of 10,
and the world model trains poorly.

Mean over the 4 tasks, seed 0:

| Regularizer | Mean return |
|---|---|
| Summed (paper, literal) | 428 |
| Averaged over dimensions | 624 |
| None | 700 |

The run without the regularizer was the best, so it was used for all four seeds.

`train_act_diag.yml` tests one more hypothesis: that Mamba sees 32-step windows in training but whole
500-step episodes when acting. The beliefs do differ, but acting with a sliding 32-step window did
not close the gap.

### Fig. 5: sequence length

Noisy walker-walk, seed 0, KalMamba without the regularizer. Values are the mean of the last 3
evaluations, with wall-clock hours in parentheses.

| Sequence length | KalMamba | VRKN | RSSM |
|---|---|---|---|
| 32 | 875 (4.7 h) | 962 (9.3 h) | 958 (7.9 h) |
| 64 | 949 (5.4 h) | 966 (19.0 h) | 952 (14.5 h) |
| 128 | 958 (5.3 h) | 951 (0.6M steps at 19.5 h) | 956 (0.78M steps at 19.4 h) |
| 256 | 774 (5.9 h) | 897 (0.3M steps at 19.7 h) | 864 (0.4M steps at 19.7 h) |

KalMamba's training time barely changes with sequence length, while VRKN and RSSM slow down and do
not reach 1M steps within the 24 h limit at lengths 128 and 256. This matches the paper.

KalMamba's return improves from length 32 to 128, but drops at 256. This is a single seed, and the
paper reports no drop. Plot: `experiments/outputs/fig5_seq_len/fig5.png`.

### Fig. 3 / 7 / 8: image tasks

Seed-0 runs are in progress; job ids are in `experiments/outputs/image_jobs.txt`. Here `kalmamba`
uses the paper's α = 1. On images the reconstruction term is about 11,000 nats, so the summed
regularizer (20 to 50) no longer dominates.

## Assumptions

The paper leaves some settings unstated. Every assumption is marked where it is set, in
`experiments/configs/kalmamba/noisy_state.yml` and `image.yml`. The main ones:

| Setting | Our choice | Source |
|---|---|---|
| Action repeat | 2 for all tasks | CoRAL locomotion config |
| Updates per episode | 250 | CoRAL locomotion config |
| Batch size | 32 sequences of length 32 | Sequence length from the paper, batch from CoRAL |
| World-model gradient clipping | Norm 10 | CoRAL |
| Noise level 0.3 | A standard deviation | Assumed |
| Transition noise | Added to the action inside the environment | Assumed |
| Regularizer target in Eq. 2 | Treated as a constant (gradient stopped) | Assumed |
| Initial KL weight | 0 | VRKN code |
| SAC temperature | Initial value 0.1 (the paper's "initial learning rate 0.1") | Assumed reading of Table 3 |
| Acting | Policy acts on the filtered mean, with no sampling | Assumed |
| Image size | 64 × 64 RGB | CoRAL / Dreamer |
| Image encoder and decoder | Dreamer conv nets with ReLU | Paper Table 2, implemented as in CoRAL |
