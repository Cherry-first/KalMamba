"""Runtime of KalMamba, RSSM and VRKN against sequence length (paper Figure 6).

Builds each method exactly as the noisy walker-walk experiment does (experiments/configs/kalmamba/
noisy_state.yml, batch size 32) and times on random data:
  - "forward": the SSM forward pass, i.e. encoder plus inferring the training-time beliefs (filtered and
    smoothed for KalMamba / VRKN, filtered for the RSSM), without gradients;
  - "batch":   one full training step on one batch: world-model loss, backward and update, then the SAC
    critic, actor and entropy updates.
Results go to a CSV; plot with scripts/plot_runtime.py.
    python scripts/bench_runtime.py --out experiments/outputs/runtime/fig6.csv
"""
import argparse
import copy
import csv
import os
import tempfile
import time

import torch
import yaml

from experiments.run_rl import Cw2MBRLExperiment

CONFIG = os.path.join(os.path.dirname(__file__), "..", "experiments", "configs", "kalmamba", "noisy_state.yml")
METHODS = {"KalMamba": "kalmamba_noreg", "RSSM": "rssm", "VRKN": "vrkn"}


def _merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in update.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def build(experiment_name: str, log_dir: str):
    docs = {d["name"]: d for d in yaml.safe_load_all(open(CONFIG))}
    params = _merge(docs["DEFAULT"]["params"], docs[experiment_name].get("params", {}))
    exp = Cw2MBRLExperiment()
    exp._printer_logger = None  # Cw2MBRLExperiment skips Cw2Experiment.__init__, which sets this
    exp.initialize({"_rep_log_path": log_dir, "params": params}, rep=0, logger=[])
    return exp._experiment, params


def random_batch(batch_size: int, seq_len: int, obs_dim: int, act_dim: int, device):
    obs = [torch.randn(batch_size, seq_len, obs_dim, device=device)]
    actions = torch.rand(batch_size, seq_len, act_dim, device=device) * 2 - 1
    rewards = torch.rand(batch_size, seq_len, 1, device=device)
    flags = torch.zeros(batch_size, seq_len, 1, dtype=torch.bool, device=device)
    return obs, actions, rewards, flags, flags, {}


def timeit(fn, warmup: int, reps: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", default="experiments/outputs/runtime/fig6.csv")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    for label in args.methods:
        with tempfile.TemporaryDirectory() as log_dir:
            experiment, params = build(METHODS[label], log_dir)
            trainer, model = experiment._trainer, experiment._model
            env = experiment._env
            obs_dim, act_dim = env.observation_space[0].shape[0], env.action_space.shape[0]
            batch_size = params["rl"]["rl_exp"]["model_updt_batch_size"]
            device = next(model.parameters()).device
            for seq_len in args.lengths:
                batch = random_batch(batch_size, seq_len, obs_dim, act_dim, device)

                def forward():
                    with torch.no_grad():
                        trainer.model_objective.compute_states(*batch, smoothed_states_if_avail=True)

                def train_step():
                    trainer._train_on_batch(batch, env_step=0, mode=None)

                reps = args.reps if seq_len <= 512 else max(2, args.reps // 2)
                t_fwd = timeit(forward, args.warmup, reps)
                t_batch = timeit(train_step, args.warmup, reps)
                rows.append({"method": label, "seq_len": seq_len, "forward_s": t_fwd, "batch_s": t_batch})
                print(f"{label:9s} T={seq_len:5d} forward {t_fwd:8.4f}s  batch {t_batch:8.4f}s", flush=True)
                with open(args.out, "w", newline="") as f:  # rewrite after every point, so a timeout keeps results
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            del experiment, trainer, model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
