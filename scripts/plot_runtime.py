"""Plot the Figure 6 runtime benchmark (scripts/bench_runtime.py output).
    python scripts/plot_runtime.py experiments/outputs/runtime/fig6.csv experiments/outputs/runtime/fig6.png
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed categorical order, so each method keeps its color in every figure.
COLORS = {"KalMamba": "#2a78d6", "RSSM": "#eb6834", "VRKN": "#1baf7a"}
LABEL_DY = {"KalMamba": 0, "RSSM": -6, "VRKN": 6}  # points; RSSM and VRKN end close together
PANELS = [("forward_s", "SSM forward pass"), ("batch_s", "Single training batch")]


def main(csv_path: str, out_path: str):
    rows = list(csv.DictReader(open(csv_path)))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (key, title) in zip(axes, PANELS):
        for method, color in COLORS.items():
            pts = sorted((int(r["seq_len"]), float(r[key])) for r in rows if r["method"] == method)
            if not pts:
                continue
            x, y = zip(*pts)
            ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5, label=method)
            ax.annotate(method, (x[-1], y[-1]), xytext=(4, LABEL_DY[method]), textcoords="offset points",
                        va="center", fontsize=9, color="#3b3b38")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Sequence length")
        ax.set_title(title, fontsize=11)
        ax.grid(True, which="major", color="#e4e3dd", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Time [s]")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
