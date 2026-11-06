# Textile anomaly detection under a memory budget

I compared three ways of compressing a patch-feature memory: greedy farthest-point selection, MiniBatchKMeans centres and random sampling. All use the same DINOv2 features and cosine nearest-neighbour image score. This directory contains the primary experiment matching the latest manuscript, together with saved scores and a CPU analysis entry point.

## Main experiment

- 279 normal training images; 190 test images (59 normal, 131 defective).
- DINOv2 ViT-B/14, 518 × 518 input, final-block spatial tokens with CLS removed.
- 1,369 L2-normalised, 768-dimensional float32 vectors per image.
- Full memory: 381,951 vectors. Compact memories: 1,024 vectors.
- Image score: mean of the largest `max(1, floor(0.01 P))` patch distances, hence **13** for `P = 1,369`.
- Compact methods use selection seeds 42, 123, 456, 789 and 1024.

| Memory | Stored features (MB) | Image AUROC |
| --- | ---: | ---: |
| Full, 381,951 entries | 1,173.353472 | 0.9445 |
| Greedy, 1,024 entries | 3.145728 | 0.9505 ± 0.0039 |
| K-means, 1,024 entries | 3.145728 | 0.9241 ± 0.0030 |
| Random, 1,024 entries | 3.145728 | 0.8838 ± 0.0194 |

Values after ± are population standard deviations across five selection seeds; the full bank is fixed. MB means decimal megabytes of the stored feature matrix, not peak GPU memory.

The greedy−k-means AUROC difference is 0.0264, with a paired image-bootstrap 95% interval of [0.0054, 0.0522]. The greedy−full interval is [−0.0077, 0.0217]; it does not establish equivalence or superiority to the full bank. These intervals hold the training set and split fixed. The image split has not been established as independent by physical sample or production batch, and selecting a preferred budget from these comparisons still needs validation on new data.

![Primary accuracy and runtime comparisons](examples/primary_results.png)

## Check the saved results on a CPU

From the repository root:

```sh
python -m pip install -r textile_anomaly_detection/requirements.txt
python textile_anomaly_detection/analyze_results.py
python textile_anomaly_detection/plot_results.py
```

The analysis checks sample IDs and labels against the manifest, reconstructs every image score from its top 13 patch distances, and recomputes AUROC and AP. It then draws 2,000 stratified image-bootstrap samples using seed 20261022. Each sampled set is shared across methods and selection seeds; seed-level AUROCs are averaged within a replicate. The pair-count implementation gives the same bootstrap statistic as the original analysis script.

Generated reports go to `reports/`. The bundled `saved_results/` files are read-only inputs to this analysis.

## Files and result provenance

| File | Purpose |
| --- | --- |
| `extract_patch_features.py` | Read the image ZIP and manifest, check image hashes, and cache CLS-free DINOv2 patches |
| `evaluate_patch_memory.py` | Build and score full/compact memories; save aligned image and patch scores |
| `analyze_results.py` | Recompute the bundled primary comparison and paired uncertainty on a CPU |
| `benchmark_end_to_end.py` | Time shared feature extraction and separate full/compact retrieval |
| `plot_results.py` | Draw the primary comparison from recomputed metrics and archived timing |
| `saved_results/manifest.csv` | Original image split, archive-relative filenames, dimensions and hashes; validation rows are retained |
| `saved_results/results/` | Full-bank result plus three methods × five seeds at 1,024 entries: JSON metrics and NPZ scores |
| `saved_results/feature_protocol.json` | Saved model, checkpoint hash, preprocessing and software/hardware versions |
| `saved_results/end_to_end_benchmark.json` | Original timing measurements, image IDs and protocol |

The archived results come from `run_20261022_v1`. JSON/NPZ files, the manifest and feature protocol were copied without changing their contents. The GPU scripts were adapted from that run's source by replacing local environment/dependency paths with command-line arguments. Their primary numerical calculations are retained. The portable timing script omits the separate CLS/global-token ablation; its measurements remain in the original archived JSON. `analyze_results.py` is a focused analysis of the bundled 1,024-entry comparison.

The root-level [`memory_compression.py`](../memory_compression.py) is a smaller algorithm example. Its NumPy scoring routine is separate from the GPU research implementation here.

## GPU computation and timing

Greedy selection and cosine retrieval use existing PyTorch CUDA operations. Retrieval chunks both query patches (512) and memory vectors (16,384), so the temporary similarity matrix stays bounded while the full feature bank remains resident on the GPU. K-means fitting uses scikit-learn on the CPU; its centres are then copied to the GPU for scoring. This work does not include custom CUDA kernels or operator-level profiler traces.

The archived timing experiment used an RTX 4060 Laptop GPU, 30 balanced test images, three passes and a discarded first pass. Greedy uses seed 42. Loading, preprocessing and DINOv2 extraction averaged 207.35 ms/image. Full and compact retrieval averaged 109.19 and 0.95465 ms/image respectively. Adding the common extraction time gives 316.54 and 208.30 ms/image: approximately 114× faster retrieval, but 1.52× faster extraction-plus-retrieval.

Per-run JSON fields such as `retrieval_ms_per_image` come from the separate cached-feature evaluation, which includes cache loading and transfer. Use `end_to_end_benchmark.json` for the shared-forward timing comparison above. Neither timing measure is an isolated CUDA-kernel latency or a complete production-service latency.

## Rerun from images

The original image ZIP, model weights and feature caches are not bundled. The source image dataset is not currently distributed through this repository. Checking the saved scores above needs none of them. A full rerun requires the matching image archive and the official DINOv2 ViT-B/14 checkpoint identified in `feature_protocol.json`.

The recorded environment was Python with PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, timm 1.0.25, NumPy 1.26.4 and Pillow 10.3.0. Install a CUDA-enabled PyTorch build matching your system; `requirements-gpu.txt` lists the remaining experiment dependencies and versions. Hardware-dependent timings and floating-point results may vary.

Create a new run directory and copy the manifest there. For example, in PowerShell from the repository root:

```powershell
New-Item -ItemType Directory -Force textile_anomaly_detection/runs/primary
Copy-Item textile_anomaly_detection/saved_results/manifest.csv textile_anomaly_detection/runs/primary/manifest.csv
python -m pip install -r textile_anomaly_detection/requirements-gpu.txt
python textile_anomaly_detection/extract_patch_features.py --run-dir textile_anomaly_detection/runs/primary --archive D:/datasets/annotations_dataset.zip --weights D:/models/dinov2_vitb14_pretrain.pth
python textile_anomaly_detection/evaluate_patch_memory.py --run-dir textile_anomaly_detection/runs/primary
python textile_anomaly_detection/benchmark_end_to_end.py --run-dir textile_anomaly_detection/runs/primary --archive D:/datasets/annotations_dataset.zip --weights D:/models/dinov2_vitb14_pretrain.pth
```

Replace the two input paths with your own. Evaluation defaults to all three compact methods, five seeds and 1,024 entries, plus the full bank. These GPU entry points have been checked for syntax and interface portability; the full image/GPU experiment has not been rerun for this repository preparation. Use a new run directory for a fresh experiment; existing result files are reused by the evaluator.

## External validation summaries

`saved_results/external_validation/` contains archived summaries from the same study, kept separate from the primary experiment:

- **MVTec AD:** 15-category macro image AUROC of 0.9881 ± 0.0014 across five seeds at 1,024 entries (`mvtec_summary.json`).
- **AITEX:** six eligible target fabrics, with greedy own-fabric macro AUROC 0.8573. The paired own-minus-other-fabric gap is 0.27625, giving cross-fabric macro AUROC about 0.5810 (`aitex_methods_summary.json`, `aitex_transfer_summary.json`).
- On target fabric 01, source-calibrated thresholds have mean false-positive rate 0.7370. This is a particular target result, not the six-target macro false-positive rate.

AITEX foreground filtering was chosen after examining initial outputs, so these are exploratory results. Source thresholds use only eight or nine held-out normal images, and threshold metrics are descriptive. This directory provides external-validation summaries, not the complete external-dataset pipelines or raw score archives; the CPU reproduction commands above cover the primary experiment only.
