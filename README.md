# SASeg: Structured Abstention Segmentation

Official implementation of **Structured Abstention Segmentation: Native Unknown Representation for Open-World Dense Prediction**.

The code trains and evaluates dense segmenters that treat the unknown as an explicit output region instead of a post-hoc thresholded score. It includes the proposed SASeg parser and the controlled baselines used in the paper.

## Highlights

- **Native unknown output space**: unknown query slots compete with known-class queries in the same region parser (Swin-T + FPN + masked Transformer decoder).
- **Prototype-sphere support**: explicit geometric support boundaries for known classes.
- **Dual protocols**: A1 (seen-unknown structured abstention) and A2 (unseen-unknown detection generalization).
- **Controlled baselines**: per-pixel K, per-pixel K+1, parameter-matched generic queries, class-agnostic queries, binary unknown branch, and a fully retrained M=0 model.
- **Memory-bounded evaluation** (`--stream-eval`) for large 3D test splits (e.g. AMOS22: 11,256 slices).

## Repository layout

```
src/saseg/
  e1_train.py              main training / evaluation entry point
  models/saseg.py          SASeg, PerPixel, ClassAgnostic, BinaryBranch
  models/factory.py        variant construction
  losses.py                known / unknown / ball / boundary / binary losses
  metrics.py               pixel, component and candidate metrics
  data/labels.py           dataset label -> protocol label mapping
  data/nifti.py            NIfTI / 2D-image / NPZ slice datasets
  tools/
    build_manifest.py      build a case-level manifest
    create_protocol.py     train/val/test protocol with leakage control
    run_matrix.py          multi-seed, multi-GPU experiment launcher
    build_experiment_log.py  aggregate metrics.json into Markdown
    analyze_candidates.py  A2 region-candidate diagnostics
    diagnose_known_threshold.py  known-mask threshold sweep
    profile_models.py      params / FLOPs / latency / memory
    make_qualitative.py    qualitative figure generation
tests/                     unit tests
scripts/                   dataset and experiment launchers
docs/EXPERIMENT_LOG.md     settings and results recorded for the paper
```

## Installation

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -c constraints.txt
pip install -e .
chmod +x scripts/*.sh
```

`requirements.txt` does not pin `torch`; install the wheel matching your CUDA driver first.

## Data preparation

The code expects a **JSON manifest** with this structure:

```json
{
  "dataset": "synapse",
  "labels": {
    "background": 0,
    "known": [1, 2, 3, 4, 5, 6, 7, 8],
    "seen_unknown": [9, 10, 11],
    "unseen_unknown": [12, 13],
    "ignore": []
  },
  "cases": [
    {
      "case_id": "img0001",
      "split": "train",
      "image": "/data/btcv/extracted/img0001.npz",
      "label": "/data/btcv/extracted/img0001.npz",
      "storage_format": "npz_2ch_zyx",
      "slice_indices": [10, 11, 12]
    }
  ]
}
```

Supported storage formats:

- `nifti_xyz`: 3D NIfTI image/label; `slice_indices` selects axial slices.
- `npz_2ch_zyx`: a `.npz` file with a `data` array of shape `(2, Z, Y, X)`; channel 0 is the image, channel 1 is the label.
- `image_2d`: a 2D PNG image/label pair (VOC2012, Cityscapes); omit `slice_indices`.

For NIfTI datasets (e.g. AMOS22):

```bash
python -m saseg.tools.build_manifest \
  --images-dir data/amos22/imagesTr \
  --labels-dir data/amos22/labelsTr \
  --output manifests/amos22.json \
  --known 1,2,3,4,6,7,8,10 \
  --seen-unknown 5,9,13,14 \
  --unseen-unknown 11,12,15
```

Create a leakage-free evaluation protocol (holds out validation cases from the official training split; optionally uses the official validation split as test):

```bash
python -m saseg.tools.create_protocol \
  --input manifests/amos22.json \
  --output protocols/amos22_seed42.json \
  --seed 42 --val-cases 20 --official-val-as-test
```

For 2D datasets, pass `--val-cases` (or `--val-fraction`) in the same way; `build_manifest` is not used because the official image lists already define the split.

Protocol conventions:

- **A1**: training and test both contain seen-unknown classes.
- **A2**: training uses seen-unknown supervision; at test time the unseen-unknown classes are mapped to the unknown output.
- A1 and A2 share the same training supervision, so an A2 run only re-encodes the evaluation splits and can reuse an A1 checkpoint (`--eval-only`).

## Training

All models are trained with 512x512 input and Swin-T (ImageNet-21k) by default.

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH

python -m saseg.e1_train \
  --manifest protocols/synapse_seed42.json \
  --variant saseg --protocol A1 \
  --output-dir runs/synapse/A1/saseg/seed42 \
  --seed 42 --epochs 100 --size 512 \
  --batch-size 1 --accumulation-steps 8 \
  --num-workers 2 --device cuda --amp --pretrained --augment
```

Variant options:

| Variant | Command |
|---|---|
| PerPixel-K | `--variant perpixel_k` |
| PerPixel-K+1 | `--variant perpixel_k1` |
| SASeg (M=2) | `--variant saseg --unknown-queries 2` |
| SASeg M=0 | `--variant saseg --unknown-queries 0` |
| Generic queries | `--variant saseg_generic --unknown-queries 2` |
| Class-agnostic queries | `--variant saseg_classagnostic` |
| Binary unknown branch | `--variant binary_branch` |

Useful options:

- `--max-steps N`: smoke tests.
- `--early-stop-patience`, `--early-stop-min-delta`: validation-based early stopping (known macro Dice).
- `--val-every`: evaluate every N epochs (use 3 for large 3D datasets).
- `--max-val-batches`: cap the per-epoch validation cost.
- `--skip-final-eval`: defer the full test evaluation (recommended for large test splits).
- `--resume last.pt`: continue an interrupted run.

## Evaluation

```bash
python -m saseg.e1_train \
  --manifest protocols/synapse_seed42.json \
  --variant saseg --unknown-queries 2 --protocol A2 \
  --eval-only --checkpoint runs/synapse/A1/saseg/seed42/best.pt \
  --output-dir runs/eval/synapse/A2/saseg/seed42 \
  --size 512 --batch-size 2 --num-workers 0 --device cuda
```

For large test splits use the memory-bounded evaluator:

```bash
python -m saseg.e1_train ... --stream-eval
```

Component metrics (connected components) can be restricted to the primary score:

```bash
python -m saseg.e1_train ... --skip-predictions --component-score unknown_union
```

Metric definitions:

- `AUROC`, `AUPR`, `FPR@TPR95`: **foreground pixels only** (`target != 0` and `!= -100`).
- `F1`, `IoU`, `Precision`, `Recall`: all pixels, threshold selected on validation and frozen.
- Component metrics: per-slice connected components, mean IoU threshold 0.1.

## Experiment matrix

`tools/run_matrix.py` launches a multi-seed matrix with GPU scheduling, resume and A2 dependencies:

```bash
python -m saseg.tools.run_matrix \
  --run-root runs/formal --dataset synapse \
  --manifest protocols/synapse_seed42.json --protocols A1,A2 \
  --models "perpixel_k=perpixel_k,perpixel_k1=perpixel_k1,saseg=saseg:2,saseg_m0=saseg:0" \
  --seeds 42,43,44 --gpus 0,1 --per-gpu 1 \
  --epochs 100 --size 512 --batch-size 1 --accumulation-steps 8 \
  --num-workers 2
```

A2 tasks reuse the A1 `best.pt` automatically. Add `--stream-eval` for large test splits and `--skip-final-eval` when the test evaluation will be run separately.

Aggregate results:

```bash
python -m saseg.tools.build_experiment_log \
  --root runs/formal_eval/synapse \
  --output runs/EXPERIMENT_LOG.md --dataset Synapse
```

## Diagnostics

```bash
# A2 candidate purity / hit rate / union IoU
python -m saseg.tools.analyze_candidates \
  --checkpoint runs/synapse/A1/saseg/seed42/best.pt \
  --manifest protocols/synapse_seed42.json \
  --variant saseg --protocol A2 --split test \
  --output runs/candidates.json

# Known-mask threshold sweep
python -m saseg.tools.diagnose_known_threshold \
  --checkpoint runs/synapse/A1/saseg/seed42/best.pt \
  --manifest protocols/synapse_seed42.json \
  --variant saseg --protocol A1 --split val \
  --thresholds 0.1,0.2,0.3,0.4,0.5 \
  --output runs/threshold_sweep.json

# Efficiency
python -m saseg.tools.profile_models --output outputs/efficiency.json \
  --variants perpixel_k,perpixel_k1,saseg --size 512
```

## Reproducibility notes

- Seeds: 42, 43, 44.
- Splits are patient/case-level; no slice from one volume appears in two splits.
- Thresholds, ensemble weights, prototype radii and early stopping use validation only.
- Each run saves `run_config.json`, `last.pt`, `best.pt`, per-epoch `metrics.json` and `final/metrics.json`.
- Test labels are never used for model selection.

## Recorded results

`docs/EXPERIMENT_LOG.md` contains the exact settings and the final Synapse / AMOS22 / VOC2012 results used in the paper, including the multi-seed mean ± std, supplementary analyses and the qualitative figure descriptions.

## License

License to be added by the authors.

## Contact

For questions about the code, please open an issue in this repository.
