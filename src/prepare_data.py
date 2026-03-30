from __future__ import annotations

import argparse

from cineseek_mm.data import download_posters, prepare_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=2000, help="Limit movies for a fast local experiment.")
    parser.add_argument("--skip-posters", action="store_true")
    parser.add_argument("--poster-workers", type=int, default=1)
    args = parser.parse_args()

    movies, _ = prepare_tables(max_items=args.max_items)
    if not args.skip_posters:
        download_posters(movies, workers=args.poster_workers)


if __name__ == "__main__":
    main()
