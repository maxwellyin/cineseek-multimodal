from __future__ import annotations

import argparse

import numpy as np

from cineseek_mm.config import (
    HYBRID_EMBEDDINGS_PATH,
    HYBRID_INDEX_PATH,
    IMAGE_EMBEDDINGS_PATH,
    IMAGE_INDEX_PATH,
    TEXT_EMBEDDINGS_PATH,
    TEXT_INDEX_PATH,
)
from cineseek_mm.encoders import fuse_embeddings, normalize_matrix
from cineseek_mm.indexing import build_ip_index, save_index


def _build_one(embeddings_path, index_path) -> None:
    embeddings = normalize_matrix(np.load(embeddings_path))
    index = build_ip_index(embeddings)
    save_index(index, index_path)
    print(f"Saved {index.ntotal} vectors to {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "image", "hybrid", "all"], default="all")
    parser.add_argument("--image-weight", type=float, default=0.35)
    args = parser.parse_args()

    if args.mode in {"text", "all"}:
        _build_one(TEXT_EMBEDDINGS_PATH, TEXT_INDEX_PATH)
    if args.mode in {"image", "all"}:
        _build_one(IMAGE_EMBEDDINGS_PATH, IMAGE_INDEX_PATH)
    if args.mode in {"hybrid", "all"}:
        text_embeddings = normalize_matrix(np.load(TEXT_EMBEDDINGS_PATH))
        image_embeddings = normalize_matrix(np.load(IMAGE_EMBEDDINGS_PATH))
        hybrid_embeddings = fuse_embeddings(text_embeddings, image_embeddings, image_weight=args.image_weight)
        np.save(HYBRID_EMBEDDINGS_PATH, hybrid_embeddings)
        index = build_ip_index(hybrid_embeddings)
        save_index(index, HYBRID_INDEX_PATH)
        print(f"Saved {index.ntotal} vectors to {HYBRID_INDEX_PATH}")


if __name__ == "__main__":
    main()
