from __future__ import annotations

from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
POSTER_DIR = DATA_DIR / "posters"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
INDEX_DIR = ARTIFACTS_DIR / "indexes"

RAW_MOVIES_PATH = RAW_DIR / "movies.csv.gz"
RAW_QUERIES_PATH = RAW_DIR / "queries.csv.gz"
ITEM_TABLE_PATH = PROCESSED_DIR / "movies.csv"
QUERY_TABLE_PATH = PROCESSED_DIR / "queries.csv"

TEXT_EMBEDDINGS_PATH = PROCESSED_DIR / "metadata_clip_embeddings.npy"
IMAGE_EMBEDDINGS_PATH = PROCESSED_DIR / "poster_clip_embeddings.npy"
HYBRID_EMBEDDINGS_PATH = PROCESSED_DIR / "hybrid_clip_embeddings.npy"
EMBEDDING_METADATA_PATH = PROCESSED_DIR / "embedding_metadata.json"

TEXT_INDEX_PATH = INDEX_DIR / "metadata.faiss"
IMAGE_INDEX_PATH = INDEX_DIR / "posters.faiss"
HYBRID_INDEX_PATH = INDEX_DIR / "hybrid.faiss"

MSRD_MOVIES_URL = "https://media.githubusercontent.com/media/metarank/msrd/master/dataset/movies.csv.gz"
MSRD_QUERIES_URL = "https://media.githubusercontent.com/media/metarank/msrd/master/dataset/queries.csv.gz"

CLIP_MODEL_NAME = os.environ.get("CINESEEK_MM_CLIP_MODEL", "openai/clip-vit-base-patch32")
RANDOM_SEED = 7


def ensure_directories() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, POSTER_DIR, INDEX_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    forced = os.environ.get("CINESEEK_MM_DEVICE")
    if forced:
        return forced
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
