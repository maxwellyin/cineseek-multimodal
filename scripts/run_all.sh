#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

python src/prepare_data.py --max-items "${MAX_ITEMS:-2000}"
python src/encode_text.py
python src/encode_image.py
python src/build_index.py --mode all
python src/evaluate.py --mode text
python src/evaluate.py --mode hybrid
