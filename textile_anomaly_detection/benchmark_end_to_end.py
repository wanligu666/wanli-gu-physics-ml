"""Time feature extraction and full/compact retrieval on 30 test images."""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import timm
    from torchvision import transforms
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    run_dir = args.run_dir
    weight_path = args.weights
    zip_path = args.archive
    model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False,
                              num_classes=0)
    incompat = model.load_state_dict(
        torch.load(weight_path, map_location="cpu", weights_only=True), strict=False
    )
    if incompat.missing_keys or incompat.unexpected_keys != ["mask_token"]:
        raise RuntimeError(f"Weight mismatch: {incompat}")
    model = model.eval().cuda()
    full_cpu = np.load(run_dir / "full_bank.npy", mmap_mode="r")
    full = torch.from_numpy(full_cpu).to("cuda")
    indices = np.load(run_dir / "results" / "greedy_seed42_indices.npy")[:1024]
    compact = full[torch.as_tensor(indices, device="cuda")]
    transform = transforms.Compose([
        transforms.Resize((518, 518)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with (run_dir / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rng = np.random.default_rng(20261022)
    chosen = []
    for split in ("test_normal", "test_defect"):
        eligible = [row for row in rows if row["split"] == split]
        idx = rng.choice(len(eligible), size=15, replace=False)
        chosen.extend(eligible[i] for i in idx)

    def query_memory(query: torch.Tensor, bank: torch.Tensor) -> float:
        pieces = []
        for q_start in range(0, 1369, 512):
            q = query[q_start:q_start + 512]
            nearest = torch.full((len(q),), -float("inf"), device="cuda")
            for m_start in range(0, len(bank), 16384):
                nearest = torch.maximum(nearest, (
                    q @ bank[m_start:m_start + 16384].T
                ).max(dim=1).values)
            pieces.append(1 - nearest)
        scores = torch.cat(pieces)
        return float(scores.topk(13).values.mean().item())

    records = []
    with zipfile.ZipFile(zip_path) as archive:
        for repetition in range(3):
            for row in chosen:
                torch.cuda.synchronize()
                start = time.perf_counter()
                with Image.open(io.BytesIO(archive.read(row["image_zip_member"]))) as image:
                    tensor = transform(image.convert("RGB")).unsqueeze(0).cuda()
                with torch.inference_mode():
                    output = model.forward_features(tensor)
                    if tuple(output.shape) != (1, 1370, 768):
                        raise RuntimeError("Unexpected token count")
                    query = F.normalize(output[:, 1:, :], dim=-1).squeeze(0)
                torch.cuda.synchronize()
                extraction_s = time.perf_counter() - start
                start = time.perf_counter()
                full_score = query_memory(query, full)
                torch.cuda.synchronize()
                full_s = time.perf_counter() - start
                start = time.perf_counter()
                compact_score = query_memory(query, compact)
                torch.cuda.synchronize()
                compact_s = time.perf_counter() - start
                records.append({
                    "repetition": repetition,
                    "image_id": row["image_id"], "label": int(row["label"]),
                    "extraction_s": extraction_s,
                    "full_retrieval_s": full_s,
                    "compact_retrieval_s": compact_s,
                    "full_score": full_score,
                    "compact_score": compact_score,
                })
            print(f"pass {repetition+1}/3 complete", flush=True)
    # Compare measured scores to the archived score arrays for all 30 IDs.
    for tag, field in (("full_381951", "full_score"),
                       ("greedy_1024_seed42", "compact_score")):
        with np.load(run_dir / "results" / f"{tag}_scores.npz") as source:
            expected = dict(zip(source["image_ids"], source["image_scores"]))
        if any(abs(item[field] - expected[item["image_id"]]) > 1e-5 for item in records):
            raise RuntimeError(f"Benchmark scoring differs from saved scores: {tag}")
    def stats(field: str) -> dict:
        values = np.array([item[field] for item in records[30:]]) * 1000
        return {"mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
                "sd_ms": float(values.std())}
    output = {
        "protocol": "30 balanced held-out images, three passes, first pass warm-up, RTX 4060; ZIP read, resize and DINOv2 included in extraction",
        "selection_seed": 20261022,
        "sample_ids": [row["image_id"] for row in chosen],
        "timing_ms": {
            "feature_extraction": stats("extraction_s"),
            "full_retrieval": stats("full_retrieval_s"),
            "greedy1024_retrieval": stats("compact_retrieval_s"),
        },
        "records": records,
    }
    extraction = output["timing_ms"]["feature_extraction"]["mean_ms"]
    output["timing_ms"]["estimated_full_end_to_end_mean_ms"] = (
        extraction + output["timing_ms"]["full_retrieval"]["mean_ms"]
    )
    output["timing_ms"]["estimated_greedy1024_end_to_end_mean_ms"] = (
        extraction + output["timing_ms"]["greedy1024_retrieval"]["mean_ms"]
    )
    (run_dir / "end_to_end_benchmark.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output["timing_ms"], indent=2))


if __name__ == "__main__":
    main()
