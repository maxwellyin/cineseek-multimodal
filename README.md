# CineSeek-MM: Multimodal Retrieval for Movie Search

CineSeek-MM extends the original CineSeek semantic movie search project into a multimodal retrieval system. It uses CLIP-style text-image embeddings over movie posters and metadata, enabling text, image, and hybrid retrieval without requiring a separate web demo.

The goal is not to build another UI. The goal is to demonstrate multimodal model engineering: encoding, indexing, evaluation, and quality/latency tradeoffs.

## System Design

```text
Text query  -> CLIP text encoder  \
                                -> shared embedding space -> FAISS -> results
Image query -> CLIP image encoder /

Movie metadata -> CLIP text encoder  \
                                      -> multimodal item representation
Movie poster   -> CLIP image encoder /
```

The project keeps the production-facing CineSeek demo separate from this technical repo:

- CineSeek: product demo, FastAPI UI, agent-assisted search.
- CineSeek-MM: technical project for multimodal retrieval, PyTorch encoding, FAISS indexing, and evaluation.

## Why Multimodal Retrieval

Text metadata captures plot, genre, cast, and tags. Posters capture visual signals that text often misses: color palette, composition, genre aesthetics, and style. Combining both gives a stronger representation for visually grounded movie search queries such as:

- `dark sci-fi movies with neon city visuals`
- `horror posters with a distorted face`
- `movies like this poster but more psychological`

## Methods

- CLIP / ViT-style image encoder for posters.
- CLIP text encoder for queries and movie metadata.
- L2-normalized shared embedding space.
- FAISS inner-product search over normalized vectors.
- Hybrid item embedding from metadata and poster vectors.
- Offline evaluation for recall@k, MRR, NDCG, and latency.

## Repository Layout

```text
cineseek-multimodal/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── posters/
├── artifacts/
│   └── indexes/
├── src/
│   ├── prepare_data.py
│   ├── encode_text.py
│   ├── encode_image.py
│   ├── build_index.py
│   ├── retrieve.py
│   ├── evaluate.py
│   └── cineseek_mm/
├── scripts/
│   └── run_all.sh
└── experiments/
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prepare data and download a poster subset:

```bash
PYTHONPATH=src python src/prepare_data.py --max-items 2000
```

Encode metadata and posters:

```bash
PYTHONPATH=src python src/encode_text.py
PYTHONPATH=src python src/encode_image.py
```

Build FAISS indexes:

```bash
PYTHONPATH=src python src/build_index.py --mode all
```

Run retrieval:

```bash
PYTHONPATH=src python src/retrieve.py --text "dark sci-fi movies with neon city visuals" --mode hybrid
PYTHONPATH=src python src/retrieve.py --image data/posters/436270.jpg --mode image
PYTHONPATH=src python src/retrieve.py --text "psychological horror" --image data/posters/882598.jpg --mode hybrid
```

Evaluate:

```bash
PYTHONPATH=src python src/evaluate.py --mode text
PYTHONPATH=src python src/evaluate.py --mode hybrid
```

Or run the full Phase 1 pipeline:

```bash
bash scripts/run_all.sh
```

## Local Qualitative Demo

CineSeek-MM includes a local-only FastAPI demo for qualitative inspection and error analysis. It is not intended for deployment.

Start it after embeddings and indexes have been built:

```bash
source .venv/bin/activate
PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  uvicorn apps.demo.app:app --host 127.0.0.1 --port 8010 --reload
```

Open:

```text
http://127.0.0.1:8010
```

Supported modes:

- `Text`: text query -> movie metadata index.
- `Image`: uploaded poster/image -> poster image index.
- `Hybrid`: text query + uploaded image -> fused query embedding.

The default hybrid image weight is `0.05`, selected by validation sweep on the original CineSeek split.

## Experiments

Recommended comparisons:

| Mode | Query | Item Representation | Purpose |
| --- | --- | --- | --- |
| text | text query | metadata text embedding | Baseline semantic retrieval |
| image | poster image | poster image embedding | Visual similarity search |
| hybrid | text / image / both | fused metadata + poster embedding | Multimodal retrieval |

Metrics:

- recall@10 / recall@50 / recall@100
- MRR
- NDCG
- average query encoding latency
- FAISS search latency
- embedding coverage for posters

## Key Findings To Report

Use this repo to produce a short results table and a few qualitative examples. The strongest narrative is:

- Multimodal retrieval improves visually grounded queries because poster embeddings capture style and aesthetics not present in metadata.
- Text-only retrieval remains strong for plot- or actor-driven queries.
- Hybrid retrieval balances semantic and visual similarity while keeping online retrieval fast through precomputed embeddings and FAISS.

## Resume Framing

**CineSeek-MM: Multimodal Retrieval System**

- Extended a semantic retrieval system to a multimodal setting using CLIP-style text-image embeddings over movie posters and metadata.
- Built a PyTorch-based pipeline for encoding, indexing, and evaluation, comparing text-only, image-only, and multimodal retrieval under latency constraints.
- Analyzed tradeoffs between embedding quality, FAISS index efficiency, poster coverage, and real-time responsiveness.

## Next Phase

Phase 2 should add lightweight training rather than just using frozen CLIP:

- Train a projection head over CLIP embeddings with MSRD relevance labels.
- Add hard-negative mining from FAISS top-k false positives.
- Compare frozen CLIP vs adapted projection head on recall@k and latency.
