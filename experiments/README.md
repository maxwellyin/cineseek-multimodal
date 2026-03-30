# Experiments

Use this directory for benchmark outputs, plots, and qualitative examples.

Phase 1 fair comparison against the original CineSeek split:

| Mode | recall@10 | recall@50 | recall@100 | MRR | NDCG | avg encode ms | avg search ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original CineSeek trained dual-tower, val | 0.528 | 0.697 | 0.755 | 0.363 | 0.443 | N/A | N/A |
| original CineSeek trained dual-tower, test | 0.537 | 0.688 | 0.747 | 0.351 | 0.433 | N/A | N/A |
| frozen CLIP text, val | 0.820 | 0.864 | 0.884 | 0.736 | 0.767 | 2.23 | 0.033 |
| frozen CLIP text, test | 0.840 | 0.881 | 0.897 | 0.747 | 0.780 | 1.90 | 0.031 |
| frozen CLIP hybrid tuned on val (`image_weight=0.05`), val | 0.821 | 0.867 | 0.883 | 0.736 | 0.768 | cached | 0.028 |
| frozen CLIP hybrid tuned on val (`image_weight=0.05`), test | 0.841 | 0.882 | 0.897 | 0.746 | 0.779 | cached | 0.028 |
| frozen CLIP hybrid, val | 0.806 | 0.862 | 0.884 | 0.702 | 0.741 | 2.04 | 0.031 |
| frozen CLIP hybrid, test | 0.827 | 0.876 | 0.891 | 0.706 | 0.747 | 1.75 | 0.031 |

Run settings:

- `candidate_items=9692`
- same original CineSeek `val_query_texts` / `test_query_texts`
- `CLIP_MODEL=openai/clip-vit-base-patch32`
- `FAISS=IndexFlatIP`
- `OMP_NUM_THREADS=1`

Interpretation:

- The earlier 2000-item result was not a fair comparison because the candidate pool was smaller.
- On the aligned 9692-item candidate pool and original split, frozen CLIP text still outperforms the trained CineSeek dual-tower checkpoint.
- The default hybrid fusion weight (`image_weight=0.35`) hurts MRR/NDCG relative to text-only, so image fusion needs validation-set tuning rather than a fixed hand-picked value.
- Validation sweep selected `image_weight=0.05`, which gives a tiny recall@10 lift over text-only but does not improve MRR/NDCG. This suggests poster features are useful only as a weak auxiliary signal for the MSRD text-query task.

Image-query poster-to-movie retrieval:

| Input variant | Label | recall@1 | recall@5 | recall@10 | MRR | NDCG | avg encode ms | avg search ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| center crop | same movie id | 0.944 | 0.980 | 0.985 | 0.959 | 0.965 | 9.15 | 0.018 |
| color jitter | same movie id | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 8.74 | 0.013 |
| blur | same movie id | 0.976 | 0.993 | 0.997 | 0.984 | 0.987 | 8.49 | 0.012 |
| thumbnail crop | same movie id | 0.780 | 0.878 | 0.908 | 0.821 | 0.842 | 8.70 | 0.011 |

This is the correct image-only task: poster image as input and movie identity as label. It is separate from the MSRD text-query relevance task.

MSRD-aligned image input:

| Split | Label setting | recall@10 | recall@50 | recall@100 | MRR | NDCG |
| --- | --- | --- | --- | --- | --- | --- |
| val | leave-one-positive-out | 0.558 | 0.689 | 0.765 | 0.233 | 0.350 |
| test | leave-one-positive-out | 0.580 | 0.699 | 0.771 | 0.245 | 0.362 |

This is the stricter image-only recommendation task: input is one relevant movie poster, and labels are the other movies in the same MSRD positive set. The inclusive variant is intentionally omitted because it is trivial: the input poster movie is itself a positive label and is usually retrieved at rank 1.

Suggested qualitative slices:

- visually grounded queries
- plot-driven queries
- genre-driven queries
- failure cases where poster style conflicts with metadata
