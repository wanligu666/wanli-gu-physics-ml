# Research Code Portfolio

This repository contains selected Python code from my undergraduate physics projects and computer-vision research. The examples cover memory-bank compression for anomaly detection, grid detection in photovoltaic images, and numerical solutions of the one-dimensional Schrödinger equation. Each sample includes tests using synthetic inputs or an analytical reference solution.

## Samples

- `memory_compression.py`: seeded farthest-point selection, MiniBatchKMeans prototypes, random sampling, cosine nearest-neighbour image scoring, and AUROC/AP calculation. PyTorch accelerates farthest-point selection when installed; NumPy is used otherwise.
- `grid_line_detection.py`: Hough-line extraction, DBSCAN merging, spacing filters, and grid-size inference for regular image structures. OpenCV is needed only for Hough-line extraction; install it with `python -m pip install opencv-python` if you use that function.
- `schrodinger_solver.py`: an RK4 shooting solver for the one-dimensional time-independent Schrödinger equation, with secant root-finding for eigenenergies and an analytical infinite-square-well benchmark. This sample is based on a second-year computational-physics assignment.

These are standalone extracts rather than complete experimental pipelines. Project datasets and saved experimental results are not included.

## Run the tests

Python 3.10 or newer is recommended.

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

The tests check selection budgets, deterministic sampling, score aggregation, metric calculation, line merging, grid inference, RK4 integration, secant convergence, and agreement with the analytical infinite-square-well energy.
