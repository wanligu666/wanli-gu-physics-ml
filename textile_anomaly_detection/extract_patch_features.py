"""Extract CLS-free DINOv2 patches using the archived image split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--splits", nargs="+",
                        default=["train_normal", "test_normal", "test_defect"])
    args = parser.parse_args()
    import numpy as np
    import PIL

    import torch
    import timm
    from torchvision import transforms

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required for the patch rerun")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    zip_path = args.archive
    run_dir = args.run_dir
    weight_path = args.weights
    cache_dir = run_dir / "patch_features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    wanted_splits = set(args.splits)

    model_name = "vit_base_patch14_dinov2.lvd142m"
    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    weights = torch.load(weight_path, map_location="cpu", weights_only=True)
    incompat = model.load_state_dict(weights, strict=False)
    if incompat.missing_keys or incompat.unexpected_keys != ["mask_token"]:
        raise RuntimeError(f"Unexpected checkpoint/model mismatch: {incompat}")
    if model.num_prefix_tokens != 1 or model.patch_embed.grid_size != (37, 37):
        raise RuntimeError("Unexpected DINOv2 token layout")
    del weights
    model = model.eval().cuda()
    transform = transforms.Compose(
        [
            transforms.Resize((518, 518)),  # PIL bilinear, matching the archived extractor
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    with (run_dir / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] in wanted_splits]
    metadata = {
        "model": model_name,
        "weight_sha256": sha256_file(weight_path),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "device": torch.cuda.get_device_name(0),
        "image_resize": "518x518 torchvision Resize with PIL bilinear",
        "normalization": "ImageNet mean/std",
        "output": "final-block 1369 spatial tokens only, L2-normalized float32",
        "selected_splits": sorted(wanted_splits),
        "manifest_images": len(rows),
    }
    (run_dir / "feature_protocol.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)

    with zipfile.ZipFile(zip_path) as archive:
        for index, row in enumerate(rows, 1):
            out_path = cache_dir / f"{row['image_id']}.npy"
            if out_path.exists():
                cached = np.load(out_path, mmap_mode="r")
                if cached.shape == (1369, 768) and cached.dtype == np.float32:
                    continue
                raise RuntimeError(f"Invalid existing feature cache: {out_path}")
            start = time.perf_counter()
            raw = archive.read(row["image_zip_member"])
            if hashlib.sha256(raw).hexdigest() != row["image_sha256"]:
                raise RuntimeError(f"JPEG changed: {row['image_id']}")
            with PIL.Image.open(io.BytesIO(raw)) as image:
                if image.size != (int(row["width_xml"]), int(row["height_xml"])):
                    raise RuntimeError(f"JPEG/XML dimensions differ: {row['image_id']}")
                tensor = transform(image.convert("RGB")).unsqueeze(0).cuda()
            with torch.inference_mode():
                all_tokens = model.forward_features(tensor)
                if tuple(all_tokens.shape) != (1, 1370, 768):
                    raise RuntimeError(f"Unexpected token shape: {tuple(all_tokens.shape)}")
                patches = torch.nn.functional.normalize(all_tokens[:, 1:, :], p=2, dim=-1)
            data = patches.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
            tmp_path = out_path.with_suffix(".tmp")
            with tmp_path.open("wb") as stream:
                np.save(stream, data)
            os.replace(tmp_path, out_path)
            if index % 10 == 0 or index == len(rows):
                print(
                    f"features {index}/{len(rows)} {row['split']} "
                    f"{time.perf_counter()-start:.2f}s/image",
                    flush=True,
                )


if __name__ == "__main__":
    main()
