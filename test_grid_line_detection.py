import numpy as np
import pytest

from grid_line_detection import cluster_and_filter_lines, detect_hough_lines, infer_grid


def _horizontal(position, length=700, start=0):
    return {"pos": position, "length": length, "x1": start, "x2": start + length}


def test_merge_nearby_lines():
    lines = [
        _horizontal(100, length=500),
        _horizontal(102, length=650),
        _horizontal(200),
        _horizontal(300),
        _horizontal(900),
    ]
    positions = cluster_and_filter_lines(
        lines, image_dimension=1000, span_dimension=1000, axis="horizontal"
    )
    assert positions == [102.0, 200.0, 300.0]


def test_grid_dimensions():
    grid = infer_grid([0, 100, 200], [0, 200, 400, 600], width=600, height=200)
    assert grid["rows"] == 2
    assert grid["cols"] == 3
    assert grid["cell_aspect_ratio"] == 2.0
    assert grid["total_cells"] == 6


@pytest.mark.parametrize("noise_position", [0, 900])
def test_edge_noise(noise_position):
    lines = [_horizontal(pos) for pos in [400, 500, 600, noise_position]]

    positions = cluster_and_filter_lines(
        lines, image_dimension=1000, span_dimension=1000, axis="horizontal"
    )

    assert positions == [400.0, 500.0, 600.0]


def test_extra_line():
    lines = [_horizontal(pos) for pos in [100, 130, 200, 300, 400]]

    positions = cluster_and_filter_lines(
        lines, image_dimension=1000, span_dimension=1000, axis="horizontal"
    )

    assert positions == [100.0, 200.0, 300.0, 400.0]


def test_vertical_span():
    lines = [
        {"pos": 100, "length": 700, "y1": 0, "y2": 700},
        {"pos": 200, "length": 700, "y1": 0, "y2": 700},
        {"pos": 300, "length": 100, "y1": 0, "y2": 100},
    ]

    positions = cluster_and_filter_lines(
        lines, image_dimension=1000, span_dimension=1000, axis="vertical"
    )

    assert positions == [100.0, 200.0]


@pytest.mark.parametrize(
    "horizontal, vertical, expected_ratio",
    [
        ([], [], 180 / 250),
        ([0, 100, 200], [], 180 / 100),
        ([], [0, 120, 240, 360], 120 / 250),
    ],
)
def test_fallback_sizes(horizontal, vertical, expected_ratio):
    grid = infer_grid(horizontal, vertical, width=540, height=500)

    assert grid["rows"] == 2
    assert grid["cols"] == 3
    assert grid["cell_aspect_ratio"] == pytest.approx(expected_ratio)
    assert grid["total_cells"] == 6


@pytest.mark.parametrize("float_scale", [1.0, 255.0])
def test_float_uint8_match(float_scale):
    pytest.importorskip("cv2")
    image = np.full((400, 600), 20, dtype=np.uint8)
    for y in [50, 150, 250, 350]:
        image[y:y + 3, :] = 200
    for x in [50, 200, 350, 500]:
        image[:, x:x + 3] = 200
    float_image = image.astype(np.float32) / 255 * float_scale

    expected = detect_hough_lines(image)

    assert expected[0] and expected[1]
    assert detect_hough_lines(float_image) == expected
