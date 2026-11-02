#!/usr/bin/env bash
# Formal experiment matrix launcher.
#
# Usage:
#   DATASET=synapse MANIFEST=protocols/synapse_seed42.json \
#   GPUS=0,1 bash scripts/run_formal_all.sh
#
# Defaults are memory-safe for shared nodes: small per-worker volume caches,
# one or two workers, and single-threaded CPU ops.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATASET=${DATASET:?set DATASET (e.g. synapse, amos22, voc2012)}
MANIFEST=${MANIFEST:?set MANIFEST to the protocol json}
GPU_LIST=${GPUS:-0}
PER_GPU=${PER_GPU:-1}
SEEDS=${SEEDS:-42,43,44}
EPOCHS=${EPOCHS:-100}
SIZE=${SIZE:-512}
BATCH=${BATCH:-1}
ACCUM=${ACCUM:-8}
WORKERS=${WORKERS:-2}
CACHE=${CACHE:-1}
CACHE_MB=${CACHE_MB:-256}
VAL_EVERY=${VAL_EVERY:-1}
MODELS=${MODELS:-"perpixel_k=perpixel_k,perpixel_k1=perpixel_k1,saseg=saseg:2,saseg_m0=saseg:0,saseg_generic=saseg_generic:2,saseg_classagnostic=saseg_classagnostic:2,binary_branch=binary_branch"}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/formal}
LOG_DIR=${LOG_DIR:-$ROOT/logs}

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export SASEG_TORCH_THREADS=${SASEG_TORCH_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

mkdir -p "$RUN_ROOT" "$LOG_DIR"
LOG="$LOG_DIR/formal_${DATASET}.log"
echo "[formal] dataset=$DATASET gpus=$GPU_LIST per_gpu=$PER_GPU seeds=$SEEDS $(date -Is)" | tee -a "$LOG"
python -m saseg.tools.run_matrix \
  --run-root "$RUN_ROOT" --dataset "$DATASET" --manifest "$MANIFEST" \
  --protocols A1,A2 --models "$MODELS" --seeds "$SEEDS" \
  --gpus "$GPU_LIST" --per-gpu "$PER_GPU" --min-free-mb 3000 \
  --epochs "$EPOCHS" --size "$SIZE" --batch-size "$BATCH" \
  --accumulation-steps "$ACCUM" --num-workers "$WORKERS" \
  --volume-cache "$CACHE" --volume-cache-mb "$CACHE_MB" \
  --val-every "$VAL_EVERY" --stagger-seconds "${STAGGER:-120}" 2>&1 | tee -a "$LOG"
