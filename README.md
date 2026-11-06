# Physics and Machine Learning Projects

Selected work from my physics degree and research on visual defect detection. Each project below has its own scope, inputs and results.

| Project | Start here | Included work |
| --- | --- | --- |
| Textile anomaly detection | [Project README](textile_anomaly_detection/README.md) | DINOv2 patch memories, three compression methods, five-seed results, paired bootstrap and GPU timing |
| Compton scattering | [Experiment README](compton_scattering/README.md) | Angle-configurable fitting notebook, measured signal/background spectra and example plots |
| Photovoltaic grid detection | [grid_line_detection.py](grid_line_detection.py) | Hough-line detection, DBSCAN merging and cell-layout inference |
| Schrödinger equation | [schrodinger_solver.py](schrodinger_solver.py) | RK4 shooting and secant root-finding, with grid and energy-tolerance comparisons |

## Textile research and the shorter algorithm sample

The [textile experiment directory](textile_anomaly_detection/) contains the primary experiment matching the latest manuscript: 279 normal training images, 381,951 patch vectors and 190 test images. Saved scores let the main statistical comparison be checked on a CPU without the original images. Feature extraction, GPU evaluation and timing have separate entry points; their inputs are described in the project README.

[memory_compression.py](memory_compression.py) is a shorter, standalone algorithm sample. It implements seeded farthest-point selection, k-means prototypes, random sampling and cosine scoring. Its image-scoring function uses NumPy; the research pipeline uses PyTorch on the GPU. The saved paper results come from that research pipeline.

## Physics and photovoltaic work

The [Compton notebook](compton_scattering/PartB_Adaptive_Fit_Angle_Input_FIXED.ipynb) fits measured scattered-photon spectra after background subtraction. Its README explains the calibration, residual diagnostics and uncertainty limits.

The photovoltaic sample estimates the cell grid in an image using classical image processing. It is one part of the wider photovoltaic inspection work; it does not contain the detector-training or U-Net pipelines. OpenCV is required for Hough-line extraction.

The Schrödinger solver is based on my second-year computational-physics assignment. Running it compares loose and tight secant tolerances so that the effect of the stopping criterion can be distinguished from spatial discretisation error.

## Run the standalone tests

Python 3.10 or newer is recommended. Each project directory lists its additional dependencies.

```sh
python -m pip install -r requirements.txt
python -m pytest -q
```

These tests cover the standalone samples. The textile README also gives commands for checking the archived experimental scores and reproducing the figures.
