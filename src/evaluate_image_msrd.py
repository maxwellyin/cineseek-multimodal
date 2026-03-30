from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cineseek_mm.config import IMAGE_INDEX_PATH
from cineseek_mm.data import load_items
from cineseek_mm.encoders import encode_images
from cineseek_mm.indexing import load_index, search
from cineseek_mm.metrics import ranking_metrics


DEFAULT_ORIGINAL_DATASET = Path("/Users/maxwellyin/Documents/GitHub/retrieval-system/data/processed/msrd_text2item_dataset.pt")


def load_aligned_original(path: Path):
    original = torch.load(path, map_location="cpu")
    items = load_items()
    original_ids = [int(original["idx_to_item_id"][idx]) for idx in range(1, original["num_items"] + 1)]
    mm_ids = items["id"].astype(int).tolist()
    if original_ids != mm_ids:
        raise RuntimeError("CineSeek-MM items are not aligned with the original CineSeek dataset.")
    return original, items


def build_image_queries(original: dict, items, split: str, input_policy: str):
    image_paths = []
    leave_one_out_positives = []
    input_rows = []

    for positive_ids in original[f"{split}_positive_ids"]:
        positive_rows = [int(item_id) - 1 for item_id in positive_ids]
        candidates = []
        for row_idx in positive_rows:
            path = Path(items.iloc[row_idx]["poster_path"])
            if path.exists() and path.stat().st_size > 0:
                candidates.append((row_idx, path))
        if not candidates:
            continue

        if input_policy == "first":
            input_row, input_path = candidates[0]
        elif input_policy == "last":
            input_row, input_path = candidates[-1]
        else:
            raise ValueError(f"Unsupported input_policy: {input_policy}")

        leave_one_out = set(positive_rows) - {input_row}
        image_paths.append(str(input_path))
        leave_one_out_positives.append(leave_one_out)
        input_rows.append(input_row)

    return image_paths, leave_one_out_positives, input_rows


def filter_non_empty(query_embeddings: np.ndarray, positives: list[set[int]], input_rows: list[int]):
    keep = [idx for idx, positive in enumerate(positives) if positive]
    return query_embeddings[keep], [positives[idx] for idx in keep], [input_rows[idx] for idx in keep]


def evaluate_rankings(index, query_embeddings: np.ndarray, positives: list[set[int]], k: int):
    search_start = time.perf_counter()
    _, idxes = search(index, query_embeddings, k=k)
    search_seconds = time.perf_counter() - search_start
    ranked = [[int(idx) for idx in row.tolist()] for row in idxes]
    metrics = ranking_metrics(ranked, positives, k_values=(10, 50, 100))
    metrics["avg_search_ms"] = 1000.0 * search_seconds / max(len(query_embeddings), 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--original-dataset", type=Path, default=DEFAULT_ORIGINAL_DATASET)
    parser.add_argument("--input-policy", choices=["first", "last"], default="first")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    original, items = load_aligned_original(args.original_dataset)
    paths, leave_one_out, input_rows = build_image_queries(
        original,
        items,
        split=args.split,
        input_policy=args.input_policy,
    )
    encode_start = time.perf_counter()
    query_embeddings, valid_mask = encode_images(paths, batch_size=args.batch_size)
    encode_seconds = time.perf_counter() - encode_start
    valid = valid_mask.tolist()
    query_embeddings = query_embeddings[valid_mask]
    leave_one_out = [positive for positive, is_valid in zip(leave_one_out, valid) if is_valid]
    input_rows = [row for row, is_valid in zip(input_rows, valid) if is_valid]

    index = load_index(IMAGE_INDEX_PATH)
    leave_embeddings, leave_positives, leave_input_rows = filter_non_empty(query_embeddings, leave_one_out, input_rows)
    leave_metrics = evaluate_rankings(index, leave_embeddings, leave_positives, k=args.k)

    leave_metrics["avg_encode_ms"] = 1000.0 * encode_seconds / max(len(query_embeddings), 1)
    leave_metrics["num_queries"] = int(len(leave_embeddings))

    payload = {
        "settings": {
            "split": args.split,
            "candidate_items": int(len(items)),
            "input": "poster from one MSRD positive movie",
            "index": "movie poster image embeddings",
            "label": "same MSRD positive set minus the input poster movie",
            "input_policy": args.input_policy,
        },
        "leave_one_positive_out": leave_metrics,
    }

    output = args.output or Path(f"experiments/image_msrd_{args.split}_{args.input_policy}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
