"""Export a static per-scene comparison from the completed saved report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("results/dense"))
    args = parser.parse_args()
    run = args.run_dir
    report = json.loads((run/"comparison.json").read_text())
    if report["status"] != "completed" or len(report["all_13"]) != 3:
        raise RuntimeError("Plot only the completed three-configuration 13-scene comparison")
    scenes = report["all_13"][0]["scenes"]
    variants = ["baseline", "maxconf", "edge_anchor_direct"]
    labels = ["Median baseline", "Highest confidence", "Reliable anchors + edges"]
    colors = ["#788594", "#2985b7", "#d47828"]
    scores = np.array([[json.loads((run/"tasks"/s/v/"result.json").read_text())["scores"]["metrics"]["fscore"][1]*100
                        for s in scenes] for v in variants])
    y = np.arange(len(scenes))
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5), gridspec_kw={"width_ratios": [1.6, 1]}, sharey=True)
    for i in range(3):
        axes[0].barh(y+(i-1)*.23, scores[i], height=.22, color=colors[i], label=labels[i])
    for i in [1, 2]:
        axes[1].barh(y+(i-1.5)*.25, scores[i]-scores[0], height=.24, color=colors[i])
    axes[0].set_yticks(y, [s+(" *" if s in ["pipes", "kicker", "office"] else "") for s in scenes])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Official F-score at 2 cm (%)")
    axes[1].set_xlabel("Change from median baseline (percentage points)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
    axes[1].axvline(0, color="#444444", linewidth=.9)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("DenseStar: fixed InfoStar q4 input, all 13 public ETH3D scan scenes", fontsize=14)
    fig.text(.08, .02, "* Development scene used for configuration selection. Other 10 scenes evaluated after freezing parameters.\n"
             "Evaluation-only camera alignment; 5 mm point sampling; official normalization voxel 1 cm. Public training data, not an independent test set.",
             fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, .07, 1, .91))
    fig.savefig(run/"f2-per-scene.png", dpi=180)
    fig.savefig(run/"f2-per-scene.pdf")
    plt.close(fig)
    print(run/"f2-per-scene.png")


if __name__ == "__main__":
    main()
