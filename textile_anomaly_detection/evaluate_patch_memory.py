"""Compare full, greedy, k-means and random patch memories on the GPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["full", "greedy", "kmeans", "random"],
                        choices=["full", "greedy", "kmeans", "random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    parser.add_argument("--budgets", nargs="+", type=int, default=[1024])
    parser.add_argument("--force-greedy-rebuild", action="store_true")
    args = parser.parse_args()
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import average_precision_score, roc_auc_score

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    run_dir = args.run_dir
    cache_dir = run_dir / "patch_features"
    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)
    with (run_dir / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    train = [r for r in manifest if r["split"] == "train_normal"]
    test = [r for r in manifest if r["split"] in ("test_normal", "test_defect")]
    assert len(train) == 279 and len(test) == 190
    if not all((cache_dir / f"{r['image_id']}.npy").exists() for r in train + test):
        raise RuntimeError("Patch cache is incomplete")
    labels = np.array([int(r["label"]) for r in test], dtype=np.int64)
    assert (labels == 0).sum() == 59 and (labels == 1).sum() == 131

    bank_path = run_dir / "full_bank.npy"
    if not bank_path.exists():
        bank = np.lib.format.open_memmap(
            bank_path, mode="w+", dtype=np.float32, shape=(len(train) * 1369, 768)
        )
        for i, row in enumerate(train):
            patches = np.load(cache_dir / f"{row['image_id']}.npy")
            if patches.shape != (1369, 768) or patches.dtype != np.float32:
                raise RuntimeError(f"Invalid patch cache: {row['image_id']}")
            if not np.isfinite(patches).all() or np.max(
                np.abs(np.linalg.norm(patches, axis=1) - 1)
            ) > 1e-5:
                raise RuntimeError(f"Unnormalised patches: {row['image_id']}")
            bank[i * 1369 : (i + 1) * 1369] = patches
        bank.flush()
        del bank
    bank = np.load(bank_path, mmap_mode="r")
    if bank.shape != (381951, 768) or bank.dtype != np.float32:
        raise RuntimeError("Full bank shape differs from clean-split protocol")
    bank_hash = sha256_file(bank_path)
    print(f"full_bank={bank.shape} sha256={bank_hash}", flush=True)
    bank_gpu = torch.from_numpy(bank).to("cuda")
    torch.cuda.synchronize()
    del bank

    def score_all(memory: torch.Tensor) -> tuple[np.ndarray, np.ndarray, float]:
        # Both dimensions are chunked, so the full bank never creates an
        # image-by-bank dense similarity matrix on the 8 GB GPU.
        image_scores = np.empty(len(test), dtype=np.float64)
        patch_scores = np.empty((len(test), 1369), dtype=np.float32)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i, row in enumerate(test):
            query_cpu = np.load(cache_dir / f"{row['image_id']}.npy")
            query = torch.from_numpy(query_cpu).to("cuda")
            best_chunks = []
            for q_start in range(0, 1369, 512):
                q = query[q_start : q_start + 512]
                best = torch.full((len(q),), -float("inf"), device="cuda")
                for m_start in range(0, len(memory), 16384):
                    similarity = q @ memory[m_start : m_start + 16384].T
                    best = torch.maximum(best, similarity.max(dim=1).values)
                best_chunks.append(1.0 - best)
            values = torch.cat(best_chunks).cpu().numpy()
            patch_scores[i] = values
            image_scores[i] = float(np.sort(values)[-13:].mean())
            if (i + 1) % 25 == 0 or i + 1 == len(test):
                print(f"  scored {i+1}/{len(test)}", flush=True)
        torch.cuda.synchronize()
        return image_scores, patch_scores, time.perf_counter() - start

    def save_result(method: str, budget: int, seed: int | None,
                    memory: torch.Tensor, build_s: float | None, extras: dict) -> None:
        tag = f"{method}_{budget}" + (f"_seed{seed}" if seed is not None else "")
        result_path = results_dir / f"{tag}.json"
        if result_path.exists():
            if method == "greedy" and build_s is not None:
                old = json.loads(result_path.read_text(encoding="utf-8"))
                old["build_s"] = build_s
                old["build_time_basis"] = "measured cumulative trajectory time"
                result_path.write_text(json.dumps(old, indent=2), encoding="utf-8")
            print(f"skip existing {tag}", flush=True)
            return
        print(f"scoring {tag}", flush=True)
        scores, patches, retrieval_s = score_all(memory)
        np.savez_compressed(
            results_dir / f"{tag}_scores.npz",
            image_ids=np.array([r["image_id"] for r in test]),
            labels=labels,
            image_scores=scores,
            patch_scores=patches,
        )
        result = {
            "protocol": "clean_279_train_cls_excluded_518_bilinear_v1",
            "method": method,
            "budget": budget,
            "seed": seed,
            "train_images": len(train),
            "test_normal": 59,
            "test_defect": 131,
            "full_memory_vectors": 381951,
            "full_bank_sha256": bank_hash,
            "stored_feature_MB": round(budget * 768 * 4 / 1e6, 6),
            "auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "build_s": build_s,
            "retrieval_190_images_s": retrieval_s,
            "retrieval_ms_per_image": retrieval_s / 190 * 1000,
            "scores_file": f"{tag}_scores.npz",
            **extras,
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)

    methods = set(args.methods)
    seeds = args.seeds
    budgets = sorted(set(args.budgets))

    if "full" in methods:
        save_result("full", len(bank_gpu), None, bank_gpu, 0.0, {})
    if "greedy" in methods:
        for seed in seeds:
            indices_path = results_dir / f"greedy_seed{seed}_indices.npy"
            times_path = results_dir / f"greedy_seed{seed}_checkpoint_times.json"
            rebuild = args.force_greedy_rebuild
            existing = np.load(indices_path) if indices_path.exists() else np.empty(0)
            if len(existing) >= max(budgets) and not rebuild:
                indices = existing
                checkpoint_times = (
                    json.loads(times_path.read_text(encoding="utf-8"))
                    if times_path.exists() else {}
                )
            else:
                rng = np.random.default_rng(seed)
                chosen = [int(rng.integers(len(bank_gpu)))]
                min_distance = torch.full((len(bank_gpu),), float("inf"), device="cuda")
                selected = torch.zeros((len(bank_gpu),), dtype=torch.bool, device="cuda")
                selected[chosen[0]] = True
                checkpoint_times = {}
                torch.cuda.synchronize()
                start = time.perf_counter()
                for i in range(1, max(budgets)):
                    dist = (1.0 - bank_gpu @ bank_gpu[chosen[-1]]).clamp_min_(0)
                    min_distance = torch.minimum(min_distance, dist)
                    min_distance.masked_fill_(selected, -1)
                    nxt = int(torch.argmax(min_distance).item())
                    chosen.append(nxt)
                    selected[nxt] = True
                    if i + 1 in budgets:
                        torch.cuda.synchronize()
                        checkpoint_times[str(i + 1)] = time.perf_counter() - start
                    if (i + 1) % 256 == 0:
                        print(f"greedy seed={seed} selected={i+1}", flush=True)
                indices = np.array(chosen, dtype=np.int64)
                if len(existing) and not np.array_equal(existing, indices[:len(existing)]):
                    raise RuntimeError("Greedy trajectory changed from the earlier saved run")
                np.save(indices_path, indices)
                times_path.write_text(json.dumps(checkpoint_times, indent=2), encoding="utf-8")
            for budget in budgets:
                compact = bank_gpu[torch.as_tensor(indices[:budget], device="cuda")]
                save_result(
                    "greedy", budget, seed, compact,
                    checkpoint_times.get(str(budget)),
                    {"selection": "nested farthest-point trajectory",
                     "max_trajectory_budget": max(budgets)},
                )
    if "random" in methods:
        for seed in seeds:
            for budget in budgets:
                start = time.perf_counter()
                indices = np.sort(np.random.default_rng(seed).choice(
                    len(bank_gpu), size=budget, replace=False
                ))
                compact = bank_gpu[torch.as_tensor(indices, device="cuda")]
                torch.cuda.synchronize()
                build_s = time.perf_counter() - start
                np.save(results_dir / f"random_{budget}_seed{seed}_indices.npy", indices)
                save_result("random", budget, seed, compact, build_s,
                            {"selection": "independent seed per budget"})
    if "kmeans" in methods:
        bank_cpu = np.load(bank_path, mmap_mode="r")
        for seed in seeds:
            for budget in budgets:
                if (results_dir / f"kmeans_{budget}_seed{seed}.json").exists():
                    print(f"skip existing kmeans_{budget}_seed{seed}", flush=True)
                    continue
                start = time.perf_counter()
                km = MiniBatchKMeans(
                    n_clusters=budget, random_state=seed, n_init=3,
                    max_iter=100, batch_size=min(4096, max(256, budget * 4)),
                )
                km.fit(bank_cpu)
                centers = km.cluster_centers_.astype(np.float32)
                centers /= np.linalg.norm(centers, axis=1, keepdims=True).clip(1e-8)
                build_s = time.perf_counter() - start
                np.save(results_dir / f"kmeans_{budget}_seed{seed}_centers.npy", centers)
                compact = torch.from_numpy(centers).to("cuda")
                save_result("kmeans", budget, seed, compact, build_s,
                            {"inertia": float(km.inertia_)})


if __name__ == "__main__":
    main()
