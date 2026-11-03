from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SERIES = [
    ("Dummy baseline", None, "Dummy", "#6B7280"),
    ("Project model selector", None, "Proposed selector", "#087E8B"),
    ("FLAML AutoML", 8.0, "FLAML (8 s)", "#E07A27"),
    ("FLAML AutoML", 60.0, "FLAML (60 s)", "#4267AC"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot five-seed model results with standard-deviation error bars.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("output/experiments/model_refresh_20260926_113918/model_summary.csv"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    output_dir = args.output_dir or args.summary.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        ("classification", ["WDBC", "Wine", "Iris"], "accuracy", "Accuracy", (0.0, 1.06)),
        ("regression", ["Diabetes", "California Housing"], "r2", "R²", (-0.08, 0.95)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)

    for ax, (task, datasets, metric, ylabel, ylim) in zip(axes, panels):
        x = np.arange(len(datasets))
        width = 0.19
        for index, (method, budget, label, color) in enumerate(SERIES):
            means, deviations = [], []
            for dataset in datasets:
                rows = summary[(summary["dataset"] == dataset) & (summary["method"] == method)]
                if budget is None:
                    rows = rows[rows["budget_seconds"].isna()]
                else:
                    rows = rows[rows["budget_seconds"] == budget]
                if len(rows) != 1:
                    raise ValueError(f"Expected one result for {dataset}, {label}; found {len(rows)}")
                means.append(float(rows.iloc[0][f"{metric}_mean"]))
                deviations.append(float(rows.iloc[0][f"{metric}_sd"]))

            offset = (index - (len(SERIES) - 1) / 2) * width
            ax.bar(
                x + offset,
                means,
                width,
                yerr=deviations,
                capsize=2.5,
                linewidth=0.45,
                edgecolor="white",
                color=color,
                label=label,
                error_kw={"elinewidth": 0.8, "capthick": 0.8},
                zorder=3,
            )

        ax.set_title(task.capitalize(), fontsize=10, fontweight="bold", pad=7)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x, datasets, fontsize=8)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#D9DEE7", linewidth=0.65, zorder=0)
        ax.tick_params(axis="y", labelsize=8, length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#AAB2BF")
        if metric == "r2":
            ax.axhline(0, color="#49515E", linewidth=0.8, zorder=2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.07), handlelength=1.4, columnspacing=1.2)
    fig.text(
        0.5,
        -0.14,
        "Bars show mean; error bars show ±1 standard deviation across five seeded splits.",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#424A57",
    )
    fig.savefig(output_dir / "fig2_predictive_performance.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "fig2_predictive_performance.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure files to {output_dir}")


if __name__ == "__main__":
    main()
