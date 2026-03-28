from __future__ import annotations

import csv
import time
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import requests
from tqdm import tqdm

from cineseek_mm.config import (
    ITEM_TABLE_PATH,
    MSRD_MOVIES_URL,
    MSRD_QUERIES_URL,
    POSTER_DIR,
    QUERY_TABLE_PATH,
    RAW_MOVIES_PATH,
    RAW_QUERIES_PATH,
    ensure_directories,
)


TEXT_COLUMNS = ["title", "overview", "tags", "genres", "director", "actors", "characters"]


def sanitize_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).replace("\n", " ").replace("\r", " ").strip().split())


def maybe_download(url: str, path: Path) -> None:
    if path.exists():
        print(f"Using existing raw file: {path}")
        return
    print(f"Downloading {url}")
    urlretrieve(url, path)
    print(f"Saved {path}")


def build_title_text(row: pd.Series) -> str:
    title = sanitize_text(row.get("title"))
    year = sanitize_text(row.get("year"))
    if year and year != "0":
        return f"{title} ({year})"
    return title


def build_metadata_text(row: pd.Series, max_chars: int = 1400) -> str:
    pieces = [
        row["title_text"],
        f"genres: {row['genres']}" if row["genres"] else "",
        f"overview: {row['overview']}" if row["overview"] else "",
        f"tags: {row['tags']}" if row["tags"] else "",
        f"director: {row['director']}" if row["director"] else "",
        f"actors: {row['actors']}" if row["actors"] else "",
        f"characters: {row['characters']}" if row["characters"] else "",
    ]
    text = " ".join(piece for piece in pieces if piece).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


def prepare_tables(max_items: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_directories()
    maybe_download(MSRD_MOVIES_URL, RAW_MOVIES_PATH)
    maybe_download(MSRD_QUERIES_URL, RAW_QUERIES_PATH)

    movies = pd.read_csv(RAW_MOVIES_PATH, sep="\t", compression="gzip", quoting=csv.QUOTE_MINIMAL)
    for column in TEXT_COLUMNS + ["poster_url"]:
        movies[column] = movies[column].map(sanitize_text)
    movies = movies[movies["title"].map(bool) & movies["poster_url"].map(bool)].copy()
    movies["title_text"] = movies.apply(build_title_text, axis=1)
    movies["metadata_text"] = movies.apply(build_metadata_text, axis=1)
    movies = movies.reset_index(drop=True)
    if max_items is not None:
        movies = movies.head(max_items).copy()
    movies["item_idx"] = range(1, len(movies) + 1)
    movies["poster_path"] = movies["id"].map(lambda movie_id: str(POSTER_DIR / f"{int(movie_id)}.jpg"))

    valid_ids = set(movies["id"].astype(int).tolist())
    queries = pd.read_csv(RAW_QUERIES_PATH, sep="\t", compression="gzip", quoting=csv.QUOTE_MINIMAL)
    queries["query"] = queries["query"].map(sanitize_text)
    queries = queries[(queries["label"] > 0) & (queries["query"].str.len() >= 3)].copy()
    queries = queries[queries["id"].astype(int).isin(valid_ids)].copy()
    queries = (
        queries.groupby("query", as_index=False)["id"]
        .agg(lambda values: sorted(set(int(value) for value in values)))
        .rename(columns={"query": "query_text", "id": "positive_movie_ids"})
    )

    movies.to_csv(ITEM_TABLE_PATH, index=False)
    queries.to_csv(QUERY_TABLE_PATH, index=False)
    print(f"Saved item table: {ITEM_TABLE_PATH} ({len(movies)} rows)")
    print(f"Saved query table: {QUERY_TABLE_PATH} ({len(queries)} rows)")
    return movies, queries


def download_posters(movies: pd.DataFrame, sleep_seconds: float = 0.02, timeout: int = 20) -> None:
    POSTER_DIR.mkdir(parents=True, exist_ok=True)
    for row in tqdm(movies.itertuples(index=False), total=len(movies), desc="Downloading posters"):
        path = Path(row.poster_path)
        if path.exists() and path.stat().st_size > 0:
            continue
        try:
            response = requests.get(row.poster_url, timeout=timeout)
            response.raise_for_status()
            path.write_bytes(response.content)
        except Exception as exc:
            print(f"Skipping poster for {row.title}: {exc}")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def load_items() -> pd.DataFrame:
    if not ITEM_TABLE_PATH.exists():
        raise FileNotFoundError(f"Missing {ITEM_TABLE_PATH}. Run prepare_data.py first.")
    return pd.read_csv(ITEM_TABLE_PATH)


def load_queries() -> pd.DataFrame:
    if not QUERY_TABLE_PATH.exists():
        raise FileNotFoundError(f"Missing {QUERY_TABLE_PATH}. Run prepare_data.py first.")
    return pd.read_csv(QUERY_TABLE_PATH)
