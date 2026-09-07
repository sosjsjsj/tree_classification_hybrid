# Tree Species Classification (HybridANISO)

Classifies tree species from LAS point clouds using a 3D voxel CNN + Transformer.

## Contents

| File | Description |
|---|---|
| `train_hybrid_aniso.py` | HybridANISO_32-32-64 training and evaluation |
| `augment_las.py` | LAS periphery augmentation |
| `checkpoints/HybridANISO_32-32-64_best.pt` | Trained weights |

## Model

`HybridCNNTransformer` — a tree point cloud is converted into anisotropic voxels
(XY 32, Z 64), a 3D CNN extracts local features, and those features are fed to a
Transformer encoder together with 128 learnable query tokens to learn global
relationships. Only the Z axis is divided more finely, because trees are tall.

- CNN channels: (32, 64, 128, 256)
- embed_dim 256 / depth 6 / heads 8 / query tokens 128
- Total parameters: 6,035,206

## Preprocessing

1. Load LAS (x, y, z, intensity)
2. Sample 4096 points with exact FPS
3. Min-max normalize (coordinates against the bbox, intensity against its max)
4. Voxelize into a (32, 32, 64) grid with 2 channels: [occupancy, mean_intensity]

Preprocessed samples are cached as npy files under `data_root/.cache` and reused
on subsequent runs.

## Dataset layout

```
data_root/
    30.단풍나무(MP)/*.las
    33.갈참나무(QUA)/*.las
    ...
```

There is no need to pre-split into train/test folders. A single seed produces the
split at run time; the per-class train/test counts are fixed in `SPLIT_COUNTS`.
Classes not listed there are split by `TRAIN_RATIO` (0.7).

## Usage

### Training

```bash
python train_hybrid_aniso.py ./tree_dataset
```

A seed is drawn at random (1-10000) on every run, then printed and saved. It
determines (1) the train/test split, (2) weight initialization and (3) batch
shuffling.

Outputs (under `data_root`):
- `result_metrics_seed<seed>.json` — accuracy, f1_macro, mcc, kappa, auc_macro_ovr
- `split_seed<seed>.json` — the train/test file lists that seed produced

### Augmentation

```bash
python augment_las.py <input.las> [--seed 42] [--outdir DIR]
```

Splits the peripheral points into 60-degree sectors around the tree center,
duplicates some of them pushed outward, adds Gaussian noise, and writes
`_aug1/_aug2/_aug3.las`.

## Training hyperparameters

- EPOCHS 100 (10 warmup epochs, LinearLR 0.1x-1.0x, then CosineAnnealingLR)
- AdamW (lr 1e-4, weight_decay 1e-4), batch_size 12, grad clip 5.0
- CrossEntropyLoss

## Requirements

```
laspy numpy torch scikit-learn tqdm
```
