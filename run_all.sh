
#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-/app/input}
OUT_ROOT=${2:-/app/output}
EPOCHS=${3:-15}
BATCH=${4:-64}
REP=${5:-tqr_tensor}

python train.py --data_root "$DATA_ROOT" --output_root "$OUT_ROOT" --dataset all --epochs "$EPOCHS" --batch_size "$BATCH" --rep "$REP"
