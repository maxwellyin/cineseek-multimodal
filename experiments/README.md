# Experiments

Use this directory for benchmark outputs, plots, and qualitative examples.

Suggested first table:

| Mode | recall@10 | recall@50 | recall@100 | MRR | NDCG | avg encode ms | avg search ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| text | 0.874 | 0.913 | 0.930 | 0.786 | 0.817 | 2.85 | 0.017 |
| hybrid | 0.875 | 0.913 | 0.933 | 0.782 | 0.814 | 2.65 | 0.018 |

Run settings:

- `MAX_ITEMS=2000`
- `max_queries=1000`
- `CLIP_MODEL=openai/clip-vit-base-patch32`
- `FAISS=IndexFlatIP`
- `OMP_NUM_THREADS=1`

Suggested qualitative slices:

- visually grounded queries
- plot-driven queries
- genre-driven queries
- failure cases where poster style conflicts with metadata
