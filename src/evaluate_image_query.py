from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from cineseek_mm.config import IMAGE_INDEX_PATH, RANDOM_SEED
from cineseek_mm.data import load_items
from cineseek_mm.encoders import encode_images
from cineseek_mm.indexing import load_index, search
from cineseek_mm.metrics import ranking_metrics


def valid_poster_rows(items, max_items: int | None, seed: int):
    rows = []
    for idx, row in items.reset_index().iterrows():
        path = Path(row["poster_path"])
        if path.exists() and path.stat().st_size > 0:
            rows.append((int(row["index"]), path))
    if max_items is not None and max_items < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, max_items)
    return rows


def augment_poster(path: Path, variant: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    width, height = image.size

    if variant == "identity":
        return image

    if variant == "center_crop":
        crop_ratio = 0.86
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        return image.crop((left, top, left + crop_width, top + crop_height)).resize((width, height))

    if variant == "color_jitter":
        image = ImageEnhance.Color(image).enhance(0.72)
        image = ImageEnhance.Contrast(image).enhance(1.18)
        return image

    if variant == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=1.2))

    if variant == "thumbnail_crop":
        crop_ratio = 0.72
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        return cropped.resize((width, height), resample=Image.Resampling.BICUBIC)

    raise ValueError(f"Unsupported variant: {variant}")


def write_augmented_images(rows: list[tuple[int, Path]], variant: str, tmpdir: Path) -> tuple[list[int], list[str]]:
    labels = []
    paths = []
    for row_idx, path in rows:
        try:
            augmented = augment_poster(path, variant)
        except Exception:
            continue
        output = tmpdir / f"{row_idx}.jpg"
        augmented.save(output, format="JPEG", quality=82)
        labels.append(row_idx)
        paths.append(str(output))
    return labels, paths


def evaluate_variant(index, rows: list[tuple[int, Path]], variant: str, batch_size: int, k: int) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        labels, paths = write_augmented_images(rows, variant, Path(tmp))
        encode_start = time.perf_counter()
        query_embeddings, valid_mask = encode_images(paths, batch_size=batch_size)
        encode_seconds = time.perf_counter() - encode_start

    labels = [label for label, valid in zip(labels, valid_mask.tolist()) if valid]
    query_embeddings = query_embeddings[valid_mask]
    positives = [{label} for label in labels]

    search_start = time.perf_counter()
    _, idxes = search(index, query_embeddings, k=k)
    search_seconds = time.perf_counter() - search_start

    ranked = [[int(idx) for idx in row.tolist()] for row in idxes]
    metrics = ranking_metrics(ranked, positives, k_values=(1, 5, 10))
    metrics.update(
        {
            "variant": variant,
            "num_queries": int(len(labels)),
            "avg_encode_ms": 1000.0 * encode_seconds / max(len(labels), 1),
            "avg_search_ms": 1000.0 * search_seconds / max(len(labels), 1),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=str, default="center_crop,color_jitter,blur,thumbnail_crop")
    parser.add_argument("--max-items", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=Path("experiments/image_query_results.json"))
    args = parser.parse_args()

    items = load_items()
    rows = valid_poster_rows(items, max_items=args.max_items, seed=args.seed)
    index = load_index(IMAGE_INDEX_PATH)
    variants = [variant.strip() for variant in args.variants.split(",") if variant.strip()]
    results = [evaluate_variant(index, rows, variant, args.batch_size, args.k) for variant in variants]

    payload = {
        "settings": {
            "candidate_items": int(len(items)),
            "sampled_image_queries": int(len(rows)),
            "label": "same movie id",
            "input": "augmented poster image",
            "index": "original poster image embeddings",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
