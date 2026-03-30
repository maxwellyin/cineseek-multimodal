from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cineseek_mm.config import (
    HYBRID_EMBEDDINGS_PATH,
    IMAGE_EMBEDDINGS_PATH,
    TEXT_EMBEDDINGS_PATH,
)
from cineseek_mm.data import load_items
from cineseek_mm.encoders import encode_texts, fuse_embeddings, normalize_matrix
from cineseek_mm.indexing import build_ip_index, search
from cineseek_mm.metrics import ranking_metrics


DEFAULT_ORIGINAL_DATASET = Path("/Users/maxwellyin/Documents/GitHub/retrieval-system/data/processed/msrd_text2item_dataset.pt")


def load_aligned_original(path: Path):
    original = torch.load(path, map_location="cpu")
    items = load_items()
    original_ids = [int(original["idx_to_item_id"][idx]) for idx in range(1, original["num_items"] + 1)]
    mm_ids = items["id"].astype(int).tolist()
    if original_ids != mm_ids:
        raise RuntimeError("CineSeek-MM items are not aligned with the original CineSeek dataset.")
    return original


def split_data(original: dict, split: str, batch_size: int):
    query_texts = list(original[f"{split}_query_texts"])
    positive_item_ids = original[f"{split}_positive_ids"]
    positives = [set(int(item_id) - 1 for item_id in ids) for ids in positive_item_ids]
    embeddings = normalize_matrix(encode_texts(query_texts, batch_size=batch_size))
    return embeddings, positives


def evaluate_weight(
    query_embeddings: np.ndarray,
    positives: list[set[int]],
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    image_weight: float,
    k: int,
) -> dict[str, float]:
    item_embeddings = fuse_embeddings(text_embeddings, image_embeddings, image_weight=image_weight)
    index = build_ip_index(item_embeddings)
    search_start = time.perf_counter()
    _, idxes = search(index, query_embeddings, k=k)
    search_seconds = time.perf_counter() - search_start
    ranked = [[int(idx) for idx in row.tolist()] for row in idxes]
    metrics = ranking_metrics(ranked, positives, k_values=(10, 50, 100))
    metrics["image_weight"] = float(image_weight)
    metrics["avg_search_ms"] = 1000.0 * search_seconds / max(len(query_embeddings), 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dataset", type=Path, default=DEFAULT_ORIGINAL_DATASET)
    parser.add_argument("--weights", type=str, default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.75,1.0")
    parser.add_argument("--select-metric", choices=["recall@10", "recall@50", "recall@100", "mrr", "ndcg"], default="recall@10")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("experiments/fusion_sweep_results.json"))
    parser.add_argument("--save-best-hybrid", action="store_true")
    args = parser.parse_args()

    original = load_aligned_original(args.original_dataset)
    text_embeddings = normalize_matrix(np.load(TEXT_EMBEDDINGS_PATH))
    image_embeddings = normalize_matrix(np.load(IMAGE_EMBEDDINGS_PATH))
    weights = [float(value.strip()) for value in args.weights.split(",") if value.strip()]

    val_query_embeddings, val_positives = split_data(original, "val", batch_size=args.batch_size)
    test_query_embeddings, test_positives = split_data(original, "test", batch_size=args.batch_size)

    val_results = [
        evaluate_weight(val_query_embeddings, val_positives, text_embeddings, image_embeddings, weight, args.k)
        for weight in weights
    ]
    best_val = max(val_results, key=lambda item: (item[args.select_metric], item["mrr"], item["ndcg"]))
    best_weight = float(best_val["image_weight"])
    test_result = evaluate_weight(test_query_embeddings, test_positives, text_embeddings, image_embeddings, best_weight, args.k)

    payload = {
        "settings": {
            "candidate_items": int(text_embeddings.shape[0]),
            "select_metric": args.select_metric,
            "weights": weights,
        },
        "best_validation": best_val,
        "test_at_best_validation_weight": test_result,
        "validation_results": val_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.save_best_hybrid:
        np.save(HYBRID_EMBEDDINGS_PATH, fuse_embeddings(text_embeddings, image_embeddings, image_weight=best_weight))

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
