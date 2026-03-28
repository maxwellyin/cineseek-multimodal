#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

python src/prepare_data.py --max-items "${MAX_ITEMS:-2000}"
python src/encode_text.py
python src/encode_image.py
python src/build_index.py --mode all
python src/evaluate.py --mode text
python src/evaluate.py --mode hybrid
