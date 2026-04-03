from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from cineseek_mm.config import (
    HYBRID_INDEX_PATH,
    IMAGE_INDEX_PATH,
    TEXT_INDEX_PATH,
)
from cineseek_mm.data import load_items
from cineseek_mm.encoders import encode_images, encode_texts, normalize_matrix
from cineseek_mm.indexing import load_index, search as search_index


@lru_cache(maxsize=1)
def load_assets():
    items = load_items()
    indexes = {
        "text": load_index(TEXT_INDEX_PATH),
        "image": load_index(IMAGE_INDEX_PATH),
        "hybrid": load_index(HYBRID_INDEX_PATH),
    }
    return items, indexes


def _query_embedding(query: str, image_path: str | None, mode: str, image_weight: float) -> np.ndarray:
    if mode == "text":
        if not query:
            raise ValueError("Text mode requires a query.")
        return normalize_matrix(encode_texts([query]))

    if mode == "image":
        if not image_path:
            raise ValueError("Image mode requires an uploaded poster/image.")
        embeddings, valid_mask = encode_images([image_path])
        if not bool(valid_mask[0]):
            raise ValueError("Could not read uploaded image.")
        return normalize_matrix(embeddings)

    if mode == "hybrid":
        if not query or not image_path:
            raise ValueError("Hybrid mode requires both a text query and an uploaded image.")
        text_embedding = normalize_matrix(encode_texts([query]))
        image_embeddings, valid_mask = encode_images([image_path])
        if not bool(valid_mask[0]):
            raise ValueError("Could not read uploaded image.")
        image_embedding = normalize_matrix(image_embeddings)
        return normalize_matrix((1.0 - image_weight) * text_embedding + image_weight * image_embedding)

    raise ValueError(f"Unsupported mode: {mode}")


def _row_to_result(row: pd.Series, score: float) -> dict[str, object]:
    return {
        "title": row["title_text"],
        "score": float(score),
        "genres": row.get("genres", ""),
        "overview": row.get("overview", ""),
        "director": row.get("director", ""),
        "actors": row.get("actors", ""),
        "poster_url": row.get("poster_url", ""),
        "metadata": row.get("metadata_text", ""),
    }


def search(query: str, image_path: str | None, mode: str, image_weight: float = 0.05, k: int = 10):
    items, indexes = load_assets()
    if mode not in indexes:
        raise ValueError(f"Unsupported mode: {mode}")
    query_embedding = _query_embedding(query, image_path, mode, image_weight)
    scores, idxes = search_index(indexes[mode], query_embedding, k=k)
    return [_row_to_result(items.iloc[int(idx)], score) for score, idx in zip(scores[0], idxes[0])]
