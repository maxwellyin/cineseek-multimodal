from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


def build_ip_index(embeddings: np.ndarray) -> faiss.Index:
    matrix = np.ascontiguousarray(embeddings.astype("float32"))
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    if not path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {path}")
    return faiss.read_index(str(path))


def search(index: faiss.Index, query_embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.ascontiguousarray(query_embeddings.astype("float32"))
    return index.search(matrix, min(k, index.ntotal))
