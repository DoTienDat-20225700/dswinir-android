#!/bin/sh
# README links under Results contain images. Only the explicit Pretrained/Model
# link below is treated as a checkpoint; it covers Rain100L SINGLE TASK.
set -eu
cd "$(dirname "$0")"
mkdir -p weights
PYTHON_BIN=${PYTHON_BIN:-python3}
"$PYTHON_BIN" -m gdown 'https://drive.google.com/uc?id=1yQcnuRUepg9NW5DVh3MmBFlA-H1r7y7E' -O weights/rain100l_official.pth
"$PYTHON_BIN" validate_model.py weights/rain100l_official.pth
