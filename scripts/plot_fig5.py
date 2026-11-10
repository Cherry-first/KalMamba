"""Plot paper Figure 5: eval return against wall-clock hours on noisy walker-walk for training sequence lengths
32/64/128/256 (runs from experiments/configs/kalmamba/fig5_seq_len.yml; works on unfinished runs too).
    python scripts/plot_fig5.py --out experiments/outputs/fig5_seq_len/fig5.png
"""
import argparse
import re

LENGTHS = [32, 64, 128, 256]
METHODS = {"kalmamba": ("KalMamba", "#2a78d6"), "rssm": ("RSSM+SAC", "#eb6834"), "vrkn": ("VRKN+SAC", "#1baf7a")}
ROW_RE = re.compile(r"Iteration +0*(\d+):.*?reward_eval/average_reward: ([0-9.]+).*?time/elapsed_h: ([0-9.]+)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", default="experiments/outputs/fig5_jobs.txt")
    parser.add_argument("--logs", default=".")
    parser.add_argument("--out", default="experiments/outputs/fig5_seq_len/fig5.png")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jobs = dict(line.split() for line in open(args.jobs) if line.strip())
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, sharey=True)
    for ax, length in zip(axes.flat, LENGTHS):
        for method, (label, color) in METHODS.items():
            job = jobs.get(f"{method}_L{length}")
            try:
                rows = ROW_RE.findall(open(f"{args.logs}/kalmamba-train-{job}.out").read())
            except (FileNotFoundError, TypeError):
                continue
            if not rows:
                continue
            steps = int(rows[-1][0]) * 1000
            hours = [float(h) for _, _, h in rows]
            ax.plot(hours, [float(r) for _, r, _ in rows], color=color, linewidth=2,
                    label=f"{label} ({steps / 1e6:.2f}M steps, {hours[-1]:.1f} h)")
        ax.set_title(f"Sequence length {length}", fontsize=11)
        ax.grid(True, color="#e4e3dd", linewidth=0.8)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Time (h)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Expected return")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
