"""Memory-bank compression and image scoring for textile anomaly detection.

Inputs are patch features extracted separately from the images.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _feature_matrix(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError("Expected a non-empty feature matrix")

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms) & (norms > 0)):
        raise ValueError("Cosine distance needs finite, non-zero features")
    return np.ascontiguousarray(array / norms)


def greedy_coreset(
    features: np.ndarray,
    budget: int,
    *,
    seed: int = 42,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Select a cosine farthest-point coreset and return vectors and indices.

    The selected set starts from a seeded random feature. PyTorch is used when
    available; the NumPy path keeps the routine usable for small CPU examples.
    """
    matrix = _feature_matrix(features)
    n_samples = len(matrix)
    if not 1 <= budget <= n_samples:
        raise ValueError("budget must be in [1, n_features]")
    if budget == n_samples:
        return matrix, np.arange(n_samples, dtype=np.int64)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(n_samples))

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device
        x = torch.as_tensor(matrix, dtype=torch.float32, device=resolved_device)
        selected = torch.empty(budget, dtype=torch.long, device=resolved_device)
        chosen = torch.zeros(n_samples, dtype=torch.bool, device=resolved_device)
        closest = torch.full((n_samples,), float("inf"), device=resolved_device)
        current = first

        with torch.inference_mode():
            for step in range(budget):
                selected[step] = current
                chosen[current] = True
                distances = torch.clamp(1.0 - x @ x[current], min=0.0, max=2.0)
                closest = torch.minimum(closest, distances)
                closest[chosen] = -1.0
                if step + 1 < budget:
                    current = int(torch.argmax(closest).item())

        indices = selected.cpu().numpy()
    else:
        if device not in {"auto", "cpu"}:
            raise ImportError("PyTorch is needed for a non-CPU device")
        indices = np.empty(budget, dtype=np.int64)
        chosen = np.zeros(n_samples, dtype=bool)
        closest = np.full(n_samples, np.inf, dtype=np.float32)
        current = first
        for step in range(budget):
            indices[step] = current
            chosen[current] = True
            distances = np.clip(1.0 - matrix @ matrix[current], 0.0, 2.0)
            np.minimum(closest, distances, out=closest)
            closest[chosen] = -1.0
            if step + 1 < budget:
                current = int(np.argmax(closest))

    return matrix[indices], indices


def kmeans_prototypes(
    features: np.ndarray,
    budget: int,
    *,
    seed: int = 42,
    n_init: int = 3,
    max_iter: int = 100,
) -> np.ndarray:
    """Use MiniBatchKMeans centres as memory vectors.

    Renormalise the centres because retrieval uses cosine distance.
    """
    from sklearn.cluster import MiniBatchKMeans

    matrix = _feature_matrix(features)
    if not 1 <= budget <= len(matrix):
        raise ValueError("budget must be in [1, n_features]")
    if budget == len(matrix):
        return matrix

    model = MiniBatchKMeans(
        n_clusters=budget,
        random_state=seed,
        batch_size=min(4096, max(256, budget * 4)),
        n_init=n_init,
        max_iter=max_iter,
    )
    centers = model.fit(matrix).cluster_centers_.astype(np.float32, copy=False)
    return _feature_matrix(centers)


def random_coreset(
    features: np.ndarray,
    budget: int,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a reproducible subset without replacement."""
    matrix = _feature_matrix(features)
    if not 1 <= budget <= len(matrix):
        raise ValueError("budget must be in [1, n_features]")
    indices = np.random.default_rng(seed).choice(len(matrix), budget, replace=False)
    return matrix[indices], indices.astype(np.int64, copy=False)


def image_scores(
    memory: np.ndarray,
    images: Sequence[np.ndarray],
    *,
    top_fraction: float = 0.01,
    patch_batch_size: int = 64,
    memory_batch_size: int = 8192,
) -> np.ndarray:
    """Average the highest patch anomaly scores for each image.

    Use max(1, floor(P * top_fraction)) patches: 13 for the paper's 1,369-patch
    images at 1%. Split retrieval into chunks to avoid a full P-by-memory matrix.
    """
    bank = _feature_matrix(memory)
    if not (0 < top_fraction <= 1 and patch_batch_size > 0 and memory_batch_size > 0):
        raise ValueError("Use a fraction in (0, 1] and positive batch sizes")

    scores: list[float] = []
    for image in images:
        patches = _feature_matrix(image)
        if patches.shape[1] != bank.shape[1]:
            raise ValueError("Patch and memory dimensions differ")

        patch_distances = np.empty(len(patches), dtype=np.float32)
        for patch_start in range(0, len(patches), patch_batch_size):
            batch = patches[patch_start : patch_start + patch_batch_size]
            best_similarity = np.full(len(batch), -np.inf, dtype=np.float32)
            for memory_start in range(0, len(bank), memory_batch_size):
                bank_batch = bank[memory_start : memory_start + memory_batch_size]
                similarities = batch @ bank_batch.T
                np.maximum(best_similarity, similarities.max(axis=1), out=best_similarity)
            patch_distances[patch_start : patch_start + len(batch)] = np.clip(
                1.0 - best_similarity, 0.0, 2.0
            )

        top_count = max(1, int(len(patch_distances) * top_fraction))
        top = np.partition(patch_distances, len(patch_distances) - top_count)[-top_count:]
        scores.append(float(top.mean()))

    return np.asarray(scores, dtype=np.float64)


def evaluate_scores(labels: Iterable[int], scores: Iterable[float]) -> dict[str, float]:
    """Image-level AUROC and AP, with 0 for normal and 1 for defective."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(list(labels))
    y_score = np.asarray(list(scores), dtype=np.float64)
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }
