from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cineseek_mm.config import HYBRID_INDEX_PATH, TEXT_INDEX_PATH
from cineseek_mm.data import load_items
from cineseek_mm.encoders import encode_texts, normalize_matrix
from cineseek_mm.indexing import load_index, search
from cineseek_mm.metrics import ranking_metrics


DEFAULT_ORIGINAL_DATASET = Path("/Users/maxwellyin/Documents/GitHub/retrieval-system/data/processed/msrd_text2item_dataset.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "hybrid"], default="text")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--original-dataset", type=Path, default=DEFAULT_ORIGINAL_DATASET)
    args = parser.parse_args()

    original = torch.load(args.original_dataset, map_location="cpu")
    items = load_items()
    original_ids = [int(original["idx_to_item_id"][idx]) for idx in range(1, original["num_items"] + 1)]
    mm_ids = items["id"].astype(int).tolist()
    if original_ids != mm_ids:
        raise RuntimeError(
            "CineSeek-MM items are not aligned with the original CineSeek dataset. "
            "Run `PYTHONPATH=src python src/prepare_data.py --max-items 9692 --skip-posters` after syncing filters."
        )

    query_texts = original[f"{args.split}_query_texts"]
    positive_item_ids = original[f"{args.split}_positive_ids"]
    positives = [set(int(item_id) - 1 for item_id in ids) for ids in positive_item_ids]

    index = load_index(TEXT_INDEX_PATH if args.mode == "text" else HYBRID_INDEX_PATH)

    encode_start = time.perf_counter()
    query_embeddings = normalize_matrix(encode_texts(list(query_texts), batch_size=args.batch_size))
    encode_seconds = time.perf_counter() - encode_start

    search_start = time.perf_counter()
    _, idxes = search(index, query_embeddings, k=args.k)
    search_seconds = time.perf_counter() - search_start

    ranked = [[int(idx) for idx in row.tolist()] for row in idxes]
    metrics = ranking_metrics(ranked, positives, k_values=(10, 50, 100))
    metrics.update(
        {
            "mode": args.mode,
            "split": args.split,
            "candidate_items": int(len(items)),
            "num_queries": int(len(query_texts)),
            "avg_encode_ms": 1000.0 * encode_seconds / max(len(query_texts), 1),
            "avg_search_ms": 1000.0 * search_seconds / max(len(query_texts), 1),
        }
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
