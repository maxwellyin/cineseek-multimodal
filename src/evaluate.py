from __future__ import annotations

import argparse
import ast
import json
import time

import numpy as np

from cineseek_mm.config import HYBRID_INDEX_PATH, TEXT_INDEX_PATH
from cineseek_mm.data import load_items, load_queries
from cineseek_mm.encoders import encode_texts, normalize_matrix
from cineseek_mm.indexing import load_index, search
from cineseek_mm.metrics import ranking_metrics


def parse_positive_ids(value) -> set[int]:
    if isinstance(value, list):
        return set(int(item) for item in value)
    return set(int(item) for item in ast.literal_eval(str(value)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "hybrid"], default="text")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    items = load_items()
    movie_id_to_row = {int(movie_id): idx for idx, movie_id in enumerate(items["id"].astype(int).tolist())}
    queries = load_queries().head(args.max_queries).copy()
    positives = []
    for value in queries["positive_movie_ids"]:
        row_ids = {movie_id_to_row[movie_id] for movie_id in parse_positive_ids(value) if movie_id in movie_id_to_row}
        positives.append(row_ids)

    index = load_index(TEXT_INDEX_PATH if args.mode == "text" else HYBRID_INDEX_PATH)

    encode_start = time.perf_counter()
    query_embeddings = normalize_matrix(encode_texts(queries["query_text"].tolist(), batch_size=args.batch_size))
    encode_seconds = time.perf_counter() - encode_start

    search_start = time.perf_counter()
    _, idxes = search(index, query_embeddings, k=args.k)
    search_seconds = time.perf_counter() - search_start
    ranked = [[int(idx) for idx in row.tolist()] for row in idxes]
    metrics = ranking_metrics(ranked, positives, k_values=(10, 50, 100))
    metrics.update(
        {
            "mode": args.mode,
            "num_queries": int(len(queries)),
            "avg_encode_ms": 1000.0 * encode_seconds / max(len(queries), 1),
            "avg_search_ms": 1000.0 * search_seconds / max(len(queries), 1),
        }
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
