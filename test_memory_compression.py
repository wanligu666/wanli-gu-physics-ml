import numpy as np
import pytest

from memory_compression import (
    evaluate_scores,
    greedy_coreset,
    image_scores,
    kmeans_prototypes,
    random_coreset,
)


def test_coreset_selection_is_seeded_and_returns_unique_indices():
    features = np.eye(8, dtype=np.float32)
    first, first_indices = greedy_coreset(features, 4, seed=7, device="cpu")
    second, second_indices = greedy_coreset(features, 4, seed=7, device="cpu")

    np.testing.assert_array_equal(first_indices, second_indices)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first_indices)) == 4
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0)


def test_random_subset_and_kmeans_prototypes_respect_budget():
    features = np.eye(8, dtype=np.float32)
    sampled, indices = random_coreset(features, 3, seed=13)
    centers = kmeans_prototypes(features, 3, seed=13, n_init=1, max_iter=10)

    assert sampled.shape == (3, 8)
    assert len(np.unique(indices)) == 3
    assert centers.shape == (3, 8)
    np.testing.assert_allclose(np.linalg.norm(centers, axis=1), 1.0, atol=1e-5)


def test_image_score_uses_cosine_nearest_neighbour_and_top_fraction():
    memory = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    image = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

    scores = image_scores(
        memory,
        [image],
        top_fraction=0.25,
        patch_batch_size=2,
        memory_batch_size=1,
    )

    np.testing.assert_allclose(scores, [1.0])


def test_paper_image_score_averages_13_patches():
    memory = np.array([[1.0, 0.0]], dtype=np.float32)
    patches = np.tile(memory, (1369, 1))
    patches[:13] = [-1.0, 0.0]
    patches[13] = [0.0, 1.0]

    scores = image_scores(memory, [patches])

    # Thirteen distances are 2; averaging a fourteenth would lower the score.
    np.testing.assert_allclose(scores, [2.0])


def test_top_fraction_keeps_at_least_one_patch():
    memory = np.array([[1.0, 0.0]], dtype=np.float32)
    patches = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)

    np.testing.assert_allclose(image_scores(memory, [patches]), [2.0])


@pytest.mark.parametrize("patch_batch, memory_batch", [(1, 1), (7, 9), (64, 8192)])
def test_chunked_scoring_matches_full_cosine_search(patch_batch, memory_batch):
    rng = np.random.default_rng(12)
    memory = rng.normal(size=(19, 7))
    images = [rng.normal(size=(23, 7)), rng.normal(size=(31, 7))]
    bank = memory / np.linalg.norm(memory, axis=1, keepdims=True)
    expected = []
    for image in images:
        patches = image / np.linalg.norm(image, axis=1, keepdims=True)
        nearest_distances = (1 - patches @ bank.T).min(axis=1)
        count = int(len(image) * 0.15)
        expected.append(np.sort(nearest_distances)[-count:].mean())

    scores = image_scores(
        memory,
        images,
        top_fraction=0.15,
        patch_batch_size=patch_batch,
        memory_batch_size=memory_batch,
    )

    np.testing.assert_allclose(scores, expected, rtol=1e-5, atol=1e-6)


def test_metrics_are_computed_from_aligned_binary_labels():
    metrics = evaluate_scores([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics == {"auroc": 1.0, "average_precision": 1.0}
