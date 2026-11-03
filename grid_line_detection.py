"""Grid detection for photovoltaic images.

The line and spacing filters assume an approximately axis-aligned, regular grid.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.cluster import DBSCAN

Axis = Literal["horizontal", "vertical"]


def detect_hough_lines(
    gray_image: np.ndarray,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Find grid lines in a grayscale image.

    Accepts 8-bit images and float images in [0, 1] or [0, 255].
    """
    import cv2

    gray = np.asarray(gray_image)
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("Expected a non-empty grayscale image")
    if gray.dtype != np.uint8:
        if np.issubdtype(gray.dtype, np.floating) and 0 <= gray.min() and gray.max() <= 1:
            gray = gray * 255
        gray = np.rint(np.clip(gray, 0, 255)).astype(np.uint8)
    height, width = gray.shape

    mean_brightness = gray.mean()
    if mean_brightness < 60:
        clip_limit = 4.0
    elif mean_brightness < 100:
        clip_limit = 3.0
    else:
        clip_limit = 2.0
    enhanced = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    # Combine edge maps so both faint and strong boundaries can contribute.
    edges = cv2.bitwise_or(cv2.Canny(blurred, 10, 30), cv2.Canny(blurred, 30, 90))
    edges = cv2.bitwise_or(edges, cv2.Canny(blurred, 50, 150))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(50, min(width, height) // 15),
        minLineLength=min(width, height) // 8,
        maxLineGap=20,
    )
    if lines is None:
        return [], []

    horizontal: list[dict[str, float]] = []
    vertical: list[dict[str, float]] = []
    for raw_line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = map(int, raw_line)
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 5 or angle > 175:
            horizontal.append(
                {"pos": (y1 + y2) / 2, "length": length,
                 "x1": min(x1, x2), "x2": max(x1, x2)}
            )
        elif 85 < angle < 95:
            vertical.append(
                {"pos": (x1 + x2) / 2, "length": length,
                 "y1": min(y1, y2), "y2": max(y1, y2)}
            )
    return horizontal, vertical


def cluster_and_filter_lines(
    lines: list[dict[str, Any]],
    *,
    image_dimension: int,
    span_dimension: int,
    axis: Axis,
    eps_ratio: float = 1 / 50,
    min_span_ratio: float = 0.4,
) -> list[float]:
    """Merge repeated detections and keep a regularly spaced sequence.

    For horizontal lines, image_dimension is height and span_dimension is width;
    reverse them for vertical lines.
    """
    span_keys = {"horizontal": ("x1", "x2"), "vertical": ("y1", "y2")}[axis]
    accepted: list[dict[str, Any]] = []
    for line in lines:
        span = float(line[span_keys[1]]) - float(line[span_keys[0]])
        if span >= span_dimension * min_span_ratio:
            accepted.append(line)
    if not accepted:
        return []

    positions = np.asarray([line["pos"] for line in accepted], dtype=np.float32).reshape(-1, 1)
    labels = DBSCAN(eps=image_dimension * eps_ratio, min_samples=1).fit_predict(positions)
    representatives = []
    for label in np.unique(labels):
        members = [accepted[i] for i, value in enumerate(labels) if value == label]
        representatives.append(max(members, key=lambda item: item["length"]))
    positions_out = sorted(float(line["pos"]) for line in representatives)

    if len(positions_out) <= 3:
        return positions_out
    median_gap = float(np.median(np.diff(positions_out)))
    min_gap, max_gap = 0.5 * median_gap, 2.5 * median_gap

    # Try each starting line: an isolated first detection should not anchor the grid.
    best_sequence = []
    for start in range(len(positions_out)):
        sequence = [positions_out[start]]
        for position in positions_out[start + 1:]:
            if min_gap <= position - sequence[-1] <= max_gap:
                sequence.append(position)
        if len(sequence) > len(best_sequence):
            best_sequence = sequence
    return best_sequence


def infer_grid(
    horizontal_lines: list[float],
    vertical_lines: list[float],
    *,
    width: int,
    height: int,
    fallback_cell_width: int = 180,
    fallback_cell_height: int = 250,
) -> dict[str, float | int]:
    """Count cells between sorted lines and estimate their width-to-height ratio.

    Fallback sizes are in pixels, so they need adjusting for other image scales.
    """
    if len(horizontal_lines) >= 2:
        rows = len(horizontal_lines) - 1
        cell_height = float(np.mean(np.diff(horizontal_lines)))
    else:
        cell_height = fallback_cell_height
        rows = max(1, round(height / cell_height))

    if len(vertical_lines) >= 2:
        cols = len(vertical_lines) - 1
        cell_width = float(np.mean(np.diff(vertical_lines)))
    else:
        cell_width = fallback_cell_width
        cols = max(1, round(width / cell_width))

    return {
        "rows": rows,
        "cols": cols,
        "cell_aspect_ratio": cell_width / cell_height,
        "total_cells": rows * cols,
    }
