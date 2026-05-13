#!/usr/bin/env python3
"""Export the exact resized images seen by a Gemma run.

The Gemma engine loads images with PIL, converts RGBA to RGB, and resizes any
image whose pixel count exceeds max_pixels. This script reproduces that path and
writes viewable PNGs into the run folder for dashboard inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def max_pixels_from_config(run_dir: Path, explicit: int | None) -> int:
    if explicit is not None:
        return explicit

    config = load_json(run_dir / "config.json")
    inference_config = config.get("model", {}).get("inference_config")
    if inference_config:
        inference_path = resolve_path(inference_config)
        inference = load_json(inference_path)
        value = inference.get("max_pixels")
        if isinstance(value, int):
            return value

    value = config.get("max_pixels")
    if isinstance(value, int):
        return value
    return 512 * 512


def resolve_image_path(raw_path: str, image_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    normalized = raw_path.replace("\\", "/")
    if normalized.startswith(str(image_root).replace("\\", "/")):
        return Path(raw_path)

    if "MindCube_image/" in normalized:
        normalized = normalized.split("MindCube_image/", 1)[1]
    elif "other_all_image/" in normalized:
        normalized = "other_all_image/" + normalized.split("other_all_image/", 1)[1]

    return image_root / normalized


def resize_like_gemma(image: Image.Image, max_pixels: int) -> Image.Image:
    if image.mode == "RGBA":
        image = image.convert("RGB")
    if max_pixels <= 0 or image.width * image.height <= max_pixels:
        return image.copy()

    resized = image.copy()
    scale = (max_pixels / float(image.width * image.height)) ** 0.5
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resized.thumbnail(new_size, Image.Resampling.LANCZOS)
    return resized


def safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{safe[:80]}_{digest}" if safe else digest


def export_images(run_dir: Path, predictions: Path, image_root: Path, max_pixels: int) -> Dict[str, Any]:
    output_root = run_dir / "model_view_images"
    output_root.mkdir(parents=True, exist_ok=True)

    examples: Dict[str, List[Dict[str, Any]]] = {}
    missing: List[Dict[str, Any]] = []
    exported_count = 0

    for row_index, item in enumerate(iter_jsonl(predictions), start=1):
        item_id = str(item.get("id") or f"row_{row_index:04d}")
        item_dir = output_root / safe_id(item_id)
        item_dir.mkdir(parents=True, exist_ok=True)
        image_records: List[Dict[str, Any]] = []

        for image_index, raw_path in enumerate(item.get("images") or [], start=1):
            source = resolve_image_path(str(raw_path), image_root)
            if not source.exists():
                record = {
                    "id": item_id,
                    "image_index": image_index,
                    "source": str(raw_path),
                    "resolved": str(source),
                }
                missing.append(record)
                image_records.append({**record, "missing": True})
                continue

            with Image.open(source) as image:
                source_size = [image.width, image.height]
                model_image = resize_like_gemma(image, max_pixels)
                model_size = [model_image.width, model_image.height]
                out_name = f"view_{image_index:02d}.png"
                out_path = item_dir / out_name
                model_image.save(out_path)

            exported_count += 1
            image_records.append(
                {
                    "image_index": image_index,
                    "source": str(raw_path),
                    "resolved": str(source),
                    "path": str(out_path.relative_to(run_dir)).replace("\\", "/"),
                    "source_size": source_size,
                    "model_size": model_size,
                    "max_pixels": max_pixels,
                    "missing": False,
                }
            )

        examples[item_id] = image_records

    manifest = {
        "max_pixels": max_pixels,
        "predictions": str(predictions.relative_to(PROJECT_ROOT)) if predictions.is_relative_to(PROJECT_ROOT) else str(predictions),
        "image_root": str(image_root),
        "exported_images": exported_count,
        "missing_images": len(missing),
        "examples": examples,
        "missing": missing,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Export model-view images for a MindCube run dashboard")
    parser.add_argument("--run-dir", required=True, help="Run folder containing predictions.jsonl and config.json")
    parser.add_argument("--predictions", help="Override predictions file")
    parser.add_argument("--image-root", default="./data/", help="Root directory used by inference")
    parser.add_argument("--max-pixels", type=int, help="Override max_pixels instead of reading inference config")
    args = parser.parse_args()

    run_dir = resolve_path(args.run_dir)
    predictions = resolve_path(args.predictions) if args.predictions else run_dir / "predictions.jsonl"
    image_root = resolve_path(args.image_root)
    max_pixels = max_pixels_from_config(run_dir, args.max_pixels)

    manifest = export_images(run_dir, predictions, image_root, max_pixels)
    print(f"Exported {manifest['exported_images']} model-view images")
    print(f"Missing {manifest['missing_images']} images")
    print(f"Max pixels: {manifest['max_pixels']}")
    print(f"Manifest: {run_dir / 'model_view_images' / 'manifest.json'}")
    return 0 if manifest["missing_images"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
