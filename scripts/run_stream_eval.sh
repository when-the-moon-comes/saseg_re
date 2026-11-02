#!/usr/bin/env bash
# Memory-bounded evaluation for one dataset.
#
# Usage:
#   MANIFEST=protocols/synapse_seed42.json \
#   CKPT_ROOT=runs/formal/synapse/A1 \
#   OUT_ROOT=runs/formal_eval/synapse \
#   MODELS="perpixel_k:perpixel_k::unknown_class,saseg:saseg:2:unknown_union" \
#   SEEDS=42,43,44 GPU=0 bash scripts/run_stream_eval.sh
#
# MODELS entries are "label:variant:unknown_queries:primary_score".
# Runs A1 and A2; skips runs without a COMPLETE marker and existing metrics.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MANIFEST=${MANIFEST:?set MANIFEST to a protocol json}
CKPT_ROOT=${CKPT_ROOT:?set CKPT_ROOT to the A1 run directory}
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT}
MODELS=${MODELS:-"perpixel_k:perpixel_k::unknown_class,saseg:saseg:2:unknown_union"}
SEEDS=${SEEDS:-42,43,44}
GPU=${GPU:-0}
SIZE=${SIZE:-512}
BATCH=${BATCH:-2}
WORKERS=${WORKERS:-0}

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export SASEG_TORCH_THREADS=${SASEG_TORCH_THREADS:-2}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

IFS=',' read -ra SEED_LIST <<< "$SEEDS"
IFS=',' read -ra MODEL_LIST <<< "$MODELS"
for spec in "${MODEL_LIST[@]}"; do
  IFS=: read -r label variant uq score <<< "$spec"
  extra=()
  [ -n "${uq:-}" ] && extra+=(--unknown-queries "$uq")
  for seed in "${SEED_LIST[@]}"; do
    ckpt="$CKPT_ROOT/$label/seed$seed/best.pt"
    if [ ! -f "$ckpt" ]; then echo "[skip] missing $ckpt"; continue; fi
    if [ ! -f "$CKPT_ROOT/$label/seed$seed/COMPLETE" ]; then
      echo "[wait] training incomplete $label seed$seed"; continue
    fi
    for protocol in A1 A2; do
      out="$OUT_ROOT/$protocol/$label/seed$seed"
      if [ -f "$out/final/metrics.json" ]; then echo "[skip] done $out"; continue; fi
      mkdir -p "$out"
      echo "[eval] $label seed$seed $protocol $(date -Is)"
      python -m saseg.e1_train \
        --manifest "$MANIFEST" --variant "$variant" "${extra[@]}" \
        --protocol "$protocol" --eval-only --checkpoint "$ckpt" \
        --output-dir "$out" --size "$SIZE" --batch-size "$BATCH" \
        --num-workers "$WORKERS" --device cuda --stream-eval --skip-components
    done
  done
done
echo "[eval] all done $(date -Is)"
