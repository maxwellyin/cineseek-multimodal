from __future__ import annotations

import argparse
import json

import numpy as np

from cineseek_mm.config import EMBEDDING_METADATA_PATH, IMAGE_EMBEDDINGS_PATH
from cineseek_mm.data import load_items
from cineseek_mm.encoders import encode_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    items = load_items()
    embeddings, valid_mask = encode_images(items["poster_path"].tolist(), batch_size=args.batch_size)
    np.save(IMAGE_EMBEDDINGS_PATH, embeddings)

    metadata = {}
    if EMBEDDING_METADATA_PATH.exists():
        metadata = json.loads(EMBEDDING_METADATA_PATH.read_text(encoding="utf-8"))
    metadata.update(
        {
            "image_embedding_shape": list(embeddings.shape),
            "poster_coverage": float(valid_mask.mean()) if len(valid_mask) else 0.0,
        }
    )
    EMBEDDING_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved image embeddings: {IMAGE_EMBEDDINGS_PATH} {embeddings.shape}")
    print(f"Poster coverage: {valid_mask.mean():.2%}")


if __name__ == "__main__":
    main()
