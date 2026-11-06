# Compton-scattering spectrum analysis

Analysis code for the Part B measurements of my PHAS0051 Compton-scattering experiment (UCL, Year 3). The experiment was done in a pair; I set up the apparatus and wrote the analysis code. For each scattering angle, the notebook subtracts a separately measured background and fits the scattered photopeak with a Gaussian on a local linear continuum.

## Files

- `PartB_Adaptive_Fit_Angle_Input_FIXED.ipynb`: the analysis notebook.
- `40_degree_20min.txt`, `50_degree_20min.txt`, `60_degree_20min.txt`: signal spectra (1,200 s each).
- `partb_background_10min.txt`: background spectrum (600 s).
- `examples/`: output plots and console logs for the three angles.

The energy calibration comes from Part A (not included): E = 2.2513 × channel − 36.87 keV.

## Run

```sh
pip install -r compton_scattering/requirements.txt
cd compton_scattering
jupyter notebook PartB_Adaptive_Fit_Angle_Input_FIXED.ipynb
```

Run the cell and enter 40, 50 or 60. The figure is saved as `partb_<angle>deg_analysis.png`.

## Method

1. Scale the background by the acquisition-time ratio, subtract it, and propagate Poisson errors from both spectra. The last channel is excluded because it contains an acquisition artefact.
2. Smooth the net spectrum and take the most prominent local maximum within ±40 channels of the channel predicted by the Compton formula. Because the search is centred on the prediction, this step locates the peak; it is not an independent test of the formula.
3. Fit the unsmoothed net counts in a window of ±k peak widths, starting at k = 1 and increasing in steps of 0.25. Each fit is checked with the reduced χ² against its 95% upper limit and with a runs test on the residual signs (p ≥ 0.05). The scan stops at the first failure after a pass and keeps the last passing window, so that the continuum is constrained by as much data as the model can still describe.

## Results (fit error only)

| Angle | Peak centre (channel) | Energy (keV) | Compton prediction (keV) | Window k |
| --- | --- | --- | --- | --- |
| 40° | 238.92 ± 0.80 | 501.0 ± 1.8 | 507.8 | 4.00\* |
| 50° | 215.77 ± 0.90 | 448.9 ± 2.0 | 452.4 | 3.25 |
| 60° | 192.73 ± 0.82 | 397.0 ± 1.8 | 401.6 | 4.00\* |

\* Reached the largest window tested; no failing window was found.

All three energies lie 3.5–7 keV below the prediction. The errors include only the fitted peak-centre uncertainty; calibration covariance, angle uncertainty and the window choice are not yet included, so the offset should not be interpreted until the full uncertainty budget is complete.

## Limitations

- The fit window is chosen from the data, so the 95% thresholds act as diagnostics rather than exact tests.
- At 40° and 60° the scan reached its limit, so the selected window is not shown to be optimal.
