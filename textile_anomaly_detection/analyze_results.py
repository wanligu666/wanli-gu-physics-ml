"""Recompute the primary comparison from saved patch scores, without a GPU."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
SEEDS = [42, 123, 456, 789, 1024]
METHODS = ["full", "greedy", "kmeans", "random"]


def load_scores(saved):
    with (saved / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["split"] in ("test_normal", "test_defect")]
    ids = np.array([row["image_id"] for row in rows])
    labels = np.array([int(row["label"]) for row in rows])
    scores = {}
    summary = []
    for method in METHODS:
        tags = (["full_381951"] if method == "full" else
                [f"{method}_1024_seed{seed}" for seed in SEEDS])
        method_scores, aucs, aps = [], [], []
        for tag in tags:
            record = json.loads((saved / "results" / f"{tag}.json").read_text(encoding="utf-8"))
            with np.load(saved / "results" / record["scores_file"]) as data:
                if not (np.array_equal(ids, data["image_ids"]) and
                        np.array_equal(labels, data["labels"])):
                    raise ValueError(f"Sample alignment changed: {tag}")
                patches = data["patch_scores"]
                if patches.shape != (len(labels), 1369):
                    raise ValueError(f"Unexpected patch layout: {tag}")
                image_scores = data["image_scores"].copy()
                top13 = np.sort(patches, axis=1)[:, -13:].mean(axis=1)
                if not np.allclose(top13, image_scores, rtol=0, atol=1e-7):
                    raise ValueError(f"Image scores do not match the top 13 patches: {tag}")
            auc = float(roc_auc_score(labels, image_scores))
            ap = float(average_precision_score(labels, image_scores))
            if abs(auc - record["auroc"]) > 1e-12 or abs(ap - record["average_precision"]) > 1e-12:
                raise ValueError(f"Saved metrics disagree with scores: {tag}")
            method_scores.append(image_scores)
            aucs.append(auc)
            aps.append(ap)
        scores[method] = np.array(method_scores)
        summary.append({
            "method": method,
            "budget": record["budget"],
            "n_seeds": len(tags),
            "auroc_mean": float(np.mean(aucs)),
            "auroc_sd_population": float(np.std(aucs)),
            "average_precision_mean": float(np.mean(aps)),
            "stored_feature_MB": record["stored_feature_MB"],
        })
    return labels, scores, summary


def paired_bootstrap(labels, scores, repeats, seed):
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    negative_weights, positive_weights = [], []
    for _ in range(repeats):
        sampled_negative = rng.choice(len(negative), len(negative), replace=True)
        sampled_positive = rng.choice(len(positive), len(positive), replace=True)
        negative_weights.append(np.bincount(sampled_negative, minlength=len(negative)))
        positive_weights.append(np.bincount(sampled_positive, minlength=len(positive)))
    negative_weights = np.array(negative_weights)
    positive_weights = np.array(positive_weights)

    # AUROC counts correctly ordered positive/negative pairs (ties count half).
    # Resample counts give the same paired bootstrap without repeatedly sorting.
    boot = {}
    for method, runs in scores.items():
        per_seed = []
        for image_scores in runs:
            difference = image_scores[positive, None] - image_scores[negative][None, :]
            pairs = (difference > 0).astype(float) + 0.5 * (difference == 0)
            values = np.sum((positive_weights @ pairs) * negative_weights, axis=1)
            per_seed.append(values / (len(positive) * len(negative)))
        boot[method] = np.mean(per_seed, axis=0)
    return boot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-dir", type=Path, default=HERE / "saved_results")
    parser.add_argument("--output-dir", type=Path, default=HERE / "reports")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261022)
    args = parser.parse_args()
    if args.bootstrap_repeats < 1:
        parser.error("--bootstrap-repeats must be positive")

    labels, scores, summary = load_scores(args.saved_dir)
    boot = paired_bootstrap(labels, scores, args.bootstrap_repeats, args.bootstrap_seed)
    means = {row["method"]: row["auroc_mean"] for row in summary}
    for row in summary:
        row["auroc_image_bootstrap_95_ci"] = np.quantile(boot[row["method"]], [.025, .975]).tolist()
    comparisons = {}
    for first, second in [("greedy", "full"), ("greedy", "kmeans"), ("greedy", "random")]:
        comparisons[f"{first}_minus_{second}"] = {
            "observed_difference": means[first] - means[second],
            "paired_image_bootstrap_95_ci": np.quantile(boot[first] - boot[second], [.025, .975]).tolist(),
        }
    output = {
        "source_run": "run_20261022_v1",
        "test_images": len(labels),
        "test_normal": int((labels == 0).sum()),
        "test_defect": int((labels == 1).sum()),
        "image_score": "Mean of the largest 13 of 1369 patch distances",
        "bootstrap": {
            "repeats": args.bootstrap_repeats, "seed": args.bootstrap_seed,
            "unit": "Image, stratified by normal/defective label",
            "pairing": "Same sampled images across all methods and selection seeds",
            "scope": "Selection-seed AUROCs averaged within each replicate; fixed training set and split",
        },
        "summary": summary,
        "comparisons": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    fields = [key for key in summary[0] if key != "auroc_image_bootstrap_95_ci"]
    with (args.output_dir / "primary_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(f"{row['method']:7s} AUROC {row['auroc_mean']:.4f} +/- {row['auroc_sd_population']:.4f}")
    for name, result in comparisons.items():
        print(name, result["paired_image_bootstrap_95_ci"])
    print("Reports:", args.output_dir)


if __name__ == "__main__":
    main()
