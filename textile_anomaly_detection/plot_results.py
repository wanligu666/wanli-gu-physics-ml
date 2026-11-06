"""Plot the primary accuracy comparison and the separate timing experiment."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=HERE / "reports/summary.json")
    parser.add_argument("--benchmark", type=Path, default=HERE / "saved_results/end_to_end_benchmark.json")
    parser.add_argument("--output", type=Path, default=HERE / "reports/primary_results.png")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))["summary"]
    timing = json.loads(args.benchmark.read_text(encoding="utf-8"))["timing_ms"]
    fig, (accuracy, runtime) = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    x = np.arange(len(summary))
    means = [row["auroc_mean"] for row in summary]
    sd = [row["auroc_sd_population"] for row in summary]
    accuracy.errorbar(x, means, yerr=sd, fmt="o", capsize=5, color="#245a81")
    accuracy.set_xticks(x, ["Full\n381,951", "Greedy\n1,024", "K-means\n1,024", "Random\n1,024"])
    accuracy.set_ylim(.84, .98)
    accuracy.set_xlim(-.4, 3.4)
    accuracy.set_ylabel("Image AUROC")
    accuracy.set_xlabel("Method and memory entries")
    accuracy.set_title("Primary test set: 190 images")
    accuracy.grid(axis="y", alpha=.25)
    for i, value in enumerate(means):
        accuracy.annotate(f"{value:.4f}", (i, value + sd[i]), xytext=(0, 8),
                          textcoords="offset points", ha="center", fontsize=9)

    extraction = timing["feature_extraction"]["mean_ms"]
    retrieval = [timing["full_retrieval"]["mean_ms"], timing["greedy1024_retrieval"]["mean_ms"]]
    runtime.bar([0, 1], [extraction, extraction], label="Loading + preprocessing + extraction", color="#98b6c8")
    runtime.bar([0, 1], retrieval, bottom=extraction, label="Retrieval", color="#db9057")
    runtime.set_xticks([0, 1], ["Full", "Greedy 1,024"])
    runtime.set_ylabel("Mean time per image (ms)")
    runtime.set_ylim(0, 400)
    runtime.set_title("Separate 30-image timing experiment")
    runtime.legend(loc="upper right", fontsize=8)
    for i, value in enumerate(retrieval):
        runtime.text(i, extraction + value + 6, f"{extraction + value:.1f} ms", ha="center", fontsize=9)
    fig.supxlabel("Accuracy: five-seed mean +/- population SD (full bank fixed). Timing: shared extraction + retrieval; seed 42.",
                  fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
