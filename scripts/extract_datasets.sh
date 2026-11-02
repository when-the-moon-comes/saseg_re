#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-$PWD/data}
LOG_DIR="$ROOT/logs/e0"
mkdir -p "$LOG_DIR" "$ROOT/data/btcv/extracted" "$ROOT/data/cityscapes"
exec > >(tee -a "$LOG_DIR/extract.log") 2>&1

echo "[extract] start $(date -Is)"
unzip -q -n "$ROOT/data/btcv/Synapse.zip" -d "$ROOT/data/btcv/extracted"
unzip -q -n "$ROOT/data/cityscapes/raw/gtFine_trainvaltest.zip" -d "$ROOT/data/cityscapes"
unzip -q -n "$ROOT/data/cityscapes/raw/leftImg8bit_trainvaltest.zip" -d "$ROOT/data/cityscapes"
echo "[extract] complete $(date -Is)"
