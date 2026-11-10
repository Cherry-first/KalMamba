"""Aggregate the noisy-state runs into the paper's Figures 4 and 9 and a final-return table.

Reads experiments/outputs/noisy_state_jobs.txt ("<method>_<task>_s<seed> <slurm job id>", the last line for
a name wins) and the eval returns in kalmamba-train-<job>.out. As in the paper (Section 5, following
Agarwal et al. 2021) it reports the mean over seeds with 95% stratified bootstrap confidence intervals:
seeds are resampled within each task, and the aggregate (Figure 4) is the mean over tasks and seeds.
Only runs that reached the last iteration are used.
    python scripts/aggregate_results.py --out experiments/outputs/noisy_state_results
"""
import argparse
import os
import re

import numpy as np

TASKS = ["cartpole-swingup", "quadruped-walk", "walker-walk", "walker-run"]
LABELS = {"kalmamba_noreg": "KalMamba", "rssm": "RSSM+SAC", "vrkn": "VRKN+SAC", "sac": "SAC",
          "kalmamba": "KalMamba (alpha=1, sum)", "kalmamba_mean": "KalMamba (alpha=1, mean)"}
# Fixed categorical order (paper legend order), so a method keeps its color in every figure.
COLORS = {"kalmamba_noreg": "#2a78d6", "rssm": "#eb6834", "vrkn": "#1baf7a", "sac": "#eda100",
          "kalmamba": "#e87ba4", "kalmamba_mean": "#6250d6"}
NAME_RE = re.compile(r"^(.+)_(" + "|".join(TASKS) + r")_s(\d+)$")
EVAL_RE = re.compile(r"Iteration +0*(\d+):.*?reward_eval/average_reward: ([0-9.]+)")


def load_runs(job_list: str, log_dir: str, last_iteration: int) -> dict:
    """Returns {(method, task, seed): (iterations, returns)} for finished runs."""
    jobs = {}
    for line in open(job_list):
        if line.strip():
            name, job = line.split()
            jobs[name] = job
    runs = {}
    for name, job in jobs.items():
        m = NAME_RE.match(name)
        path = os.path.join(log_dir, f"kalmamba-train-{job}.out")
        if m is None or not os.path.exists(path):
            continue
        points = [(int(i), float(r)) for i, r in EVAL_RE.findall(open(path).read())]
        if not points or points[-1][0] < last_iteration:
            continue
        its, rets = map(np.array, zip(*points))
        runs[(m.group(1), m.group(2), int(m.group(3)))] = (its, rets)
    return runs


def stratified_bootstrap_ci(per_task: list[np.ndarray], reps: int, rng) -> tuple:
    """per_task: one array (seeds, ...) per task. Mean over tasks and seeds, with seeds resampled within tasks."""
    point = np.mean([x.mean(0) for x in per_task], axis=0)
    samples = []
    for _ in range(reps):
        samples.append(np.mean([x[rng.integers(0, len(x), len(x))].mean(0) for x in per_task], axis=0))
    lo, hi = np.percentile(np.stack(samples), [2.5, 97.5], axis=0)
    return point, lo, hi


def final_return(rets: np.ndarray, last_k: int) -> float:
    return float(rets[-last_k:].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", default="experiments/outputs/noisy_state_jobs.txt")
    parser.add_argument("--logs", default=".")
    parser.add_argument("--methods", nargs="+", default=["kalmamba_noreg", "rssm", "vrkn", "sac"])
    parser.add_argument("--last-iteration", type=int, default=1000)
    parser.add_argument("--last-k", type=int, default=3, help="final return = mean of the last k evaluations")
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--out", default="experiments/outputs/noisy_state_results")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    runs = load_runs(args.jobs, args.logs, args.last_iteration)
    os.makedirs(args.out, exist_ok=True)

    # Final-return table: per task and aggregated, mean [95% CI] over seeds.
    lines = ["| Method | " + " | ".join(TASKS) + " | Aggregate |", "|---" * (len(TASKS) + 2) + "|"]
    curves = {}
    for method in args.methods:
        cells, per_task_final, per_task_curve = [], [], []
        for task in TASKS:
            seeds = sorted(s for (m, t, s) in runs if m == method and t == task)
            if not seeds:
                cells.append("-")
                continue
            finals = np.array([final_return(runs[(method, task, s)][1], args.last_k) for s in seeds])
            p, lo, hi = stratified_bootstrap_ci([finals], args.reps, rng)
            cells.append(f"{p:.0f} [{lo:.0f}, {hi:.0f}] (n={len(seeds)})")
            per_task_final.append(finals)
            its = runs[(method, task, seeds[0])][0]
            curve = np.stack([runs[(method, task, s)][1] for s in seeds])
            per_task_curve.append(curve)
            curves[(method, task)] = (its, *stratified_bootstrap_ci([curve], args.reps, rng), len(seeds))
        if len(per_task_final) == len(TASKS):
            p, lo, hi = stratified_bootstrap_ci(per_task_final, args.reps, rng)
            cells.append(f"{p:.0f} [{lo:.0f}, {hi:.0f}]")
            curves[(method, "aggregate")] = (its, *stratified_bootstrap_ci(per_task_curve, args.reps, rng),
                                             min(len(c) for c in per_task_curve))
        else:
            cells.append("-")
        lines.append(f"| {LABELS.get(method, method)} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(args.out, "final_returns.md"), "w") as f:
        f.write(f"Final return = mean of the last {args.last_k} evaluations (1M env steps), "
                f"mean [95% stratified bootstrap CI] over seeds.\n\n{table}\n")
    plot(curves, args.methods, args.out)


def plot(curves: dict, methods: list[str], out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def panel(ax, key_task, title):
        for method in methods:
            if (method, key_task) not in curves:
                continue
            its, mean, lo, hi, n = curves[(method, key_task)]
            x = its / 1000  # one 1000-step episode per iteration
            ax.plot(x, mean, color=COLORS[method], linewidth=2, label=f"{LABELS.get(method, method)} (n={n})")
            ax.fill_between(x, lo, hi, color=COLORS[method], alpha=0.15, linewidth=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Environment steps (millions)")
        ax.grid(True, color="#e4e3dd", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    panel(ax, "aggregate", "State-based noisy tasks (aggregate)")
    ax.set_ylabel("Expected return")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_aggregate.png"), dpi=150)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax, task in zip(axes.flat, TASKS):
        panel(ax, task, task)
    for ax in axes[:, 0]:
        ax.set_ylabel("Expected return")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 3), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))  # room for the legend below the panels
    fig.savefig(os.path.join(out_dir, "fig9_per_task.png"), dpi=150)
    print(f"wrote {out_dir}/fig4_aggregate.png and fig9_per_task.png")


if __name__ == "__main__":
    main()
