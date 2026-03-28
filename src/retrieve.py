from __future__ import annotations

import argparse

import numpy as np

from cineseek_mm.config import HYBRID_INDEX_PATH, IMAGE_INDEX_PATH, TEXT_INDEX_PATH
from cineseek_mm.data import load_items
from cineseek_mm.encoders import encode_images, encode_texts, normalize_matrix
from cineseek_mm.indexing import load_index, search


def build_query(text: str | None, image: str | None, image_weight: float) -> np.ndarray:
    text_vec = encode_texts([text])[0] if text else None
    image_vec = encode_images([image])[0][0] if image else None
    if text_vec is not None and image_vec is not None:
        return normalize_matrix(((1.0 - image_weight) * text_vec + image_weight * image_vec)[None, :])
    if text_vec is not None:
        return normalize_matrix(text_vec[None, :])
    if image_vec is not None:
        return normalize_matrix(image_vec[None, :])
    raise ValueError("Provide --text, --image, or both.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--mode", choices=["text", "image", "hybrid"], default="hybrid")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--image-weight", type=float, default=0.35)
    args = parser.parse_args()

    index_path = {"text": TEXT_INDEX_PATH, "image": IMAGE_INDEX_PATH, "hybrid": HYBRID_INDEX_PATH}[args.mode]
    index = load_index(index_path)
    items = load_items()
    query = build_query(args.text, args.image, args.image_weight)
    scores, idxes = search(index, query, k=args.k)

    for rank, (score, idx) in enumerate(zip(scores[0], idxes[0]), start=1):
        row = items.iloc[int(idx)]
        print(f"{rank:02d}. {row['title_text']} score={float(score):.4f}")
        print(f"    genres={row.get('genres', '')}")
        print(f"    poster={row.get('poster_url', '')}")


if __name__ == "__main__":
    main()
