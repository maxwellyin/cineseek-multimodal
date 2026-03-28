from __future__ import annotations

import argparse
import json

import numpy as np

from cineseek_mm.config import EMBEDDING_METADATA_PATH, TEXT_EMBEDDINGS_PATH
from cineseek_mm.data import load_items
from cineseek_mm.encoders import encode_texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    items = load_items()
    embeddings = encode_texts(items["metadata_text"].fillna("").tolist(), batch_size=args.batch_size)
    np.save(TEXT_EMBEDDINGS_PATH, embeddings)
    metadata = {
        "num_items": int(len(items)),
        "text_embedding_shape": list(embeddings.shape),
    }
    EMBEDDING_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved text embeddings: {TEXT_EMBEDDINGS_PATH} {embeddings.shape}")


if __name__ == "__main__":
    main()
