"""Summarize training logs: diagnostics at a few iterations and the evaluation returns.

    python scripts/summarize_runs.py JOBS_FILE LOG_DIR

JOBS_FILE has one "<name> <slurm job id>" per line; logs are LOG_DIR/kalmamba-train-<job id>.out.
"""
import re
import sys

KEYS = ["mahalanobis", "m_sq_err", "filter_var", "elbo_loss", "grad_norm", "kl", "obs_loss0_mse"]


def parse(path):
    rows = {}
    for line in open(path):
        m = re.search(r"Iteration\s+(\d+):", line)
        if m:
            rows[int(m.group(1))] = dict(re.findall(r"([\w/]+): ([-0-9.e+]+)", line))
    return rows


for name, job in (l.split() for l in open(sys.argv[1]) if l.strip()):
    try:
        rows = parse(f"{sys.argv[2]}/kalmamba-train-{job}.out")
    except FileNotFoundError:
        print(f"## {name}: no log")
        continue
    print(f"## {name} (job {job}, {len(rows)} iterations)")
    for it in [0, 25, 50, 100, 150]:
        if it in rows:
            r = rows[it]
            print(f"  it {it:3d} " + " ".join(f"{k}={float(r['model_train/' + k]):.4g}"
                                             for k in KEYS if 'model_train/' + k in r))
    evals = [(it, float(r["reward_eval/average_reward"])) for it, r in sorted(rows.items())
             if "reward_eval/average_reward" in r]
    print("  eval: " + " ".join(f"{it}:{v:.0f}" for it, v in evals))
