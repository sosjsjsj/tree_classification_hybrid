"""
train_hybrid_aniso.py — HybridANISO_32-32-64 (3D CNN + Transformer) tree species
classification: training and evaluation.

=== Model ===
3D voxel CNN (anisotropic voxels, 32x32x64) -> spatial token projection ->
Transformer encoder -> classifier. A tree point cloud is turned into anisotropic
voxels (XY: 32, Z: 64), the CNN extracts local features, and those features are
fed to the Transformer together with learnable query tokens to learn global
relationships.

=== Model parameters (config) ===
    kind          : hybrid (3D CNN + Transformer)
    voxel         : (32, 32, 64)  — XY 32, Z 64 (trees are tall, so only Z is finer)
    cnn channels  : (32, 64, 128, 256)
    embed_dim     : 256
    transformer depth  : 6
    transformer heads  : 8
    query tokens  : 128
    classifier    : MLP (embed_dim -> embed_dim//2 -> num_classes)
    total parameters : 6,035,206

=== Training hyperparameters ===
    EPOCHS        : 100
    WARMUP_EPOCHS : 10   (LinearLR, 0.1x -> 1.0x)
    scheduler     : CosineAnnealingLR (T_max=90, eta_min=LR*0.01), chained after warmup
    optimizer     : AdamW (lr=1e-4, weight_decay=1e-4)
    loss          : CrossEntropyLoss
    batch_size    : 12
    grad clip     : max_norm=5.0
    procedure     : backpropagate the CE loss between predicted probabilities and
                    the ground-truth label; AdamW updates the weights.

=== Preprocessing ===
    1. Load LAS (x, y, z, intensity)
    2. Sample 4096 points with exact FPS (greedy farthest point sampling)
    3. Min-max normalize (coordinates against the bbox, intensity against its max)
    4. Voxelize: project onto a (32, 32, 64) grid as 2 channels
       [occupancy, mean_intensity]

=== Dataset layout (input) ===
    data_root/
        <class_name_1>/*.las
        <class_name_2>/*.las
        ...
    Class names are taken from the folder names verbatim (e.g. "30.단풍나무(MP)").
    There is no need to pre-split into train/test folders. A single random seed
    produces the per-class split at run time; the per-class train/test counts are
    fixed in SPLIT_COUNTS.

=== Random seed ===
    A seed is drawn at random from 1-10000 on every run, then printed and saved.
    It determines (1) the train/test split, (2) weight initialization and
    (3) batch shuffling.

=== Outputs (under data_root) ===
    result_metrics_seed<seed>.json : accuracy, f1_macro, mcc, kappa, auc_macro_ovr + report
    split_seed<seed>.json          : the train/test file lists that seed produced

=== Usage ===
    python train_hybrid_aniso.py <data_root> [--device cuda] [--epochs 100]
    e.g.
    python train_hybrid_aniso.py ./tree_dataset
"""

from __future__ import annotations
import argparse
import random
import sys
import zlib
from pathlib import Path

import laspy
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                              cohen_kappa_score, confusion_matrix,
                              matthews_corrcoef, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ────────────────────────── hyperparameters ──────────────────────────
NUM_POINTS    = 4096
VOXEL_SHAPE   = (32, 32, 64)
EPOCHS        = 100
WARMUP_EPOCHS = 10
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
ETA_MIN       = LR * 0.01
BATCH_SIZE    = 12
NUM_WORKERS   = 4
GRAD_CLIP     = 5.0
TRAIN_RATIO   = 0.7          # fallback ratio for classes absent from SPLIT_COUNTS (train:test = 7:3)
SEED_MIN      = 1            # range the per-run seed is drawn from
SEED_MAX      = 10000

# Fixed per-class train/test counts {folder name: (n_train, n_test)}.
# The seed only changes *which* files go to train/test; the counts are always
# these values, independent of the seed.
# Classes whose folder name is not listed here are split by TRAIN_RATIO.
SPLIT_COUNTS: dict[str, tuple[int, int]] = {
    "30.단풍나무(MP)":   (52, 23),
    "33.갈참나무(QUA)":  (30, 14),
    "43.산딸나무(CK)":   (29, 13),
    "44.산수유(CO)":     (28, 12),
    "67.벚나무(PS)":     (18,  9),
    "70.상수리나무(QA)": (67, 30),
}

MODEL_CONFIG = dict(channels=(32, 64, 128, 256), embed_dim=256, depth=6,
                     heads=8, tokens=128, classifier_mlp=True, dropout=0.1)


# ────────────────────────── preprocessing ──────────────────────────
def load_las(path: Path) -> np.ndarray:
    las = laspy.read(path)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float32)
    intensity = np.asarray(las.intensity, dtype=np.float32).reshape(-1, 1)
    if len(xyz) == 0:
        raise ValueError(f"Empty LAS: {path}")
    return np.concatenate((xyz, intensity), 1)


def exact_fps(points: np.ndarray, count=NUM_POINTS, device=None) -> np.ndarray:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    xyz = torch.as_tensor(points[:, :3], device=device)
    n = len(xyz)
    target = min(n, count)
    norms = (xyz * xyz).sum(1)
    centroid = xyz.mean(0)
    first = ((xyz - centroid) ** 2).sum(1).argmax()
    selected = torch.empty(target, dtype=torch.long, device=device)
    distances = torch.full((n,), torch.inf, device=device)
    current = first
    for index in range(target):
        selected[index] = current
        point = xyz[current]
        candidate = norms + (point * point).sum() - 2 * torch.mv(xyz, point)
        distances.copy_(torch.minimum(distances, candidate))
        current = distances.argmax()
    indices = selected.cpu().numpy()
    if n < count:
        indices = np.resize(indices, count)
    return points[indices].copy()


def normalize_sample(points: np.ndarray, intensity_max: float) -> np.ndarray:
    result = points.astype(np.float32, copy=True)
    xyz = result[:, :3]
    minimum, maximum = xyz.min(0), xyz.max(0)
    result[:, :3] = (xyz - minimum) / np.where(maximum > minimum, maximum - minimum, 1)
    result[:, 3] /= max(float(intensity_max), 1.0)
    return result


def voxelize(points: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    width, height, depth = shape
    xyz = np.clip(points[:, :3], 0, 1)
    x = np.minimum((xyz[:, 0] * width).astype(int), width - 1)
    y = np.minimum((xyz[:, 1] * height).astype(int), height - 1)
    z = np.minimum((xyz[:, 2] * depth).astype(int), depth - 1)
    grid = np.zeros((2, depth, height, width), np.float32)
    count = np.zeros((depth, height, width), np.float32)
    np.add.at(count, (z, y, x), 1)
    np.add.at(grid[1], (z, y, x), points[:, 3])
    grid[0] = count > 0
    occupied = count > 0
    grid[1, occupied] /= count[occupied]
    return grid


def prepare_and_cache(las_path: Path, cache_dir: Path) -> Path:
    """LAS -> FPS + normalize -> npy cache. Reused if it already exists."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{zlib.crc32(str(las_path).encode()):08x}_{las_path.stem}.npy"
    out = cache_dir / key
    if not out.exists():
        raw = load_las(las_path)
        np.save(out, normalize_sample(exact_fps(raw), raw[:, 3].max()))
    return out


# ────────────────────────── model (HybridANISO_32-32-64) ──────────────────────────
class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes, channels, embed_dim, depth, heads, tokens, dropout=0.1, classifier_mlp=False):
        super().__init__()
        pool_count = len(channels) - 1
        self.cnn_layers = nn.ModuleList()
        previous = 2
        for index, channel in enumerate(channels):
            self.cnn_layers.append(nn.Sequential(
                nn.Conv3d(previous, channel, 3, padding=1, bias=False),
                nn.BatchNorm3d(channel), nn.ReLU(True),
                nn.MaxPool3d(2) if index < pool_count else nn.Identity()))
            previous = channel
        self.feature_projection = nn.Conv3d(channels[-1], embed_dim, 1)
        self.query_tokens = nn.Parameter(torch.randn(tokens, embed_dim))
        layer = nn.TransformerEncoderLayer(embed_dim, heads, embed_dim * 4, dropout,
                                            batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, depth)
        self.classifier = (
            nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim // 2),
                          nn.ReLU(True), nn.Dropout(dropout), nn.Linear(embed_dim // 2, num_classes))
            if classifier_mlp else
            nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)))

    def forward(self, voxels):
        x = voxels
        for layer in self.cnn_layers:
            x = layer(x)
        features = self.feature_projection(x).flatten(2).transpose(1, 2)
        queries = self.query_tokens.unsqueeze(0).expand(len(voxels), -1, -1)
        encoded = self.transformer_encoder(torch.cat((queries, features), 1))
        return self.classifier(encoded.mean(1))


# ────────────────────────── dataset ──────────────────────────
def scan_dataset(root: Path) -> tuple[list[str], dict[str, list[Path]]]:
    """Walk data_root/<class_name>/*.las and build the class list and per-class file lists."""
    if not root.is_dir():
        raise SystemExit(f"Data folder not found: {root}")
    names = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    files = {name: sorted((root / name).glob("*.las")) for name in names}
    classes = [name for name in names if files[name]]
    if len(classes) < 2:
        raise SystemExit(f"At least 2 class folders containing .las files are required: {root}")
    return classes, files


def split_dataset(classes: list[str], files: dict[str, list[Path]],
                  seed: int) -> tuple[list, list]:
    """Keep the per-class counts fixed; the seed only picks which files go where.

    If the folder name is in SPLIT_COUNTS, its (train, test) counts are used as-is;
    otherwise the class is split by TRAIN_RATIO (leaving at least 1 file each in
    train and test).
    """
    rng = random.Random(seed)
    train_items: list[tuple[Path, int]] = []
    test_items: list[tuple[Path, int]] = []
    for label, cls in enumerate(classes):
        paths = list(files[cls])
        rng.shuffle(paths)
        if cls in SPLIT_COUNTS:
            n_train, n_test = SPLIT_COUNTS[cls]
            if len(paths) < n_train + n_test:
                raise SystemExit(f"Not enough files for '{cls}': {len(paths)} "
                                 f"(SPLIT_COUNTS requires train {n_train} + test {n_test})")
        else:
            n_train = len(paths) if len(paths) == 1 else min(
                max(round(len(paths) * TRAIN_RATIO), 1), len(paths) - 1)
            n_test = len(paths) - n_train
        train_items += [(p, label) for p in paths[:n_train]]
        test_items += [(p, label) for p in paths[n_train:n_train + n_test]]
    return train_items, test_items


class LasDataset(Dataset):
    """Takes a list of (las_path, label), builds the npy preprocessing cache, and returns voxel tensors."""

    def __init__(self, items: list[tuple[Path, int]], cache_dir: Path):
        self.items = [(prepare_and_cache(path, cache_dir), label) for path, label in items]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        return torch.from_numpy(voxelize(np.load(path), VOXEL_SHAPE)).float(), label


# ────────────────────────── train / evaluate ──────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description="Split a <class_name>/*.las folder into train/test with a random seed and train")
    parser.add_argument("data_root", type=Path, help="data folder laid out as <class_name>/*.las")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--cache_dir", type=Path, default=None,
                        help="where to store the preprocessing cache (default: data_root/.cache)")
    args = parser.parse_args()

    seed = random.SystemRandom().randint(SEED_MIN, SEED_MAX)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache_dir = args.cache_dir or (args.data_root / ".cache")
    epochs = args.epochs
    set_seed(seed)

    classes, files = scan_dataset(args.data_root)
    train_items, test_items = split_dataset(classes, files, seed)

    print(f"seed: {seed} (random {SEED_MIN}-{SEED_MAX})")
    print(f"classes({len(classes)}): {classes}")
    for label, cls in enumerate(classes):
        n_train = sum(1 for _, l in train_items if l == label)
        n_test = sum(1 for _, l in test_items if l == label)
        source = "fixed" if cls in SPLIT_COUNTS else f"ratio {TRAIN_RATIO}"
        print(f"  - {cls}: train {n_train} / test {n_test}  ({source})")

    train_set = LasDataset(train_items, cache_dir)
    test_set = LasDataset(test_items, cache_dir)
    print(f"train: {len(train_set)} | test: {len(test_set)}")

    generator = torch.Generator().manual_seed(seed)
    loader_kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_kw)
    test_loader  = DataLoader(test_set,  shuffle=False, **loader_kw)

    model = HybridCNNTransformer(len(classes), **MODEL_CONFIG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                                total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - WARMUP_EPOCHS,
                                                         eta_min=ETA_MIN)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine],
                                                       milestones=[WARMUP_EPOCHS])
    criterion = nn.CrossEntropyLoss()

    def evaluate():
        model.eval()
        la, pa, pr = [], [], []
        with torch.no_grad():
            for inputs, target in test_loader:
                prob = torch.softmax(model(inputs.to(device)), 1).cpu().numpy()
                la.extend(target.numpy()); pa.extend(prob.argmax(1)); pr.extend(prob)
        return np.asarray(la), np.asarray(pa), np.asarray(pr)

    la = pa = proba = None
    for epoch in range(1, epochs + 1):
        model.train(); loss_sum = correct = total = 0
        bar = tqdm(train_loader, desc=f"ep={epoch}/{epochs}", leave=False)
        for inputs, labels in bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs); loss = criterion(logits, labels); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
            bar.set_postfix(loss=f"{loss_sum/total:.4f}")
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Per-epoch evaluation, for monitoring progress.
        la, pa, proba = evaluate()
        print(f"ep={epoch}/{epochs} lr={current_lr:.3e} "
              f"train_loss={loss_sum/total:.4f} train_acc={correct/total:.4f} "
              f"test_acc={accuracy_score(la, pa):.4f}")

    report = classification_report(la, pa, target_names=classes, output_dict=True, zero_division=0)
    metrics = {
        "accuracy":      float(accuracy_score(la, pa)),
        "f1_macro":      report["macro avg"]["f1-score"],
        "mcc":           float(matthews_corrcoef(la, pa)),
        "kappa":         float(cohen_kappa_score(la, pa)),
        "auc_macro_ovr": float(roc_auc_score(la, proba, multi_class="ovr", average="macro")),
        "seed":          seed,
        "total_epochs":  epochs,
        "train_samples": int(len(train_set)),
        "test_samples":  int(len(la)),
    }

    print("\n=== Results ===")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v}")
    print("\nConfusion matrix:")
    print(confusion_matrix(la, pa))

    import json
    split_path = args.data_root / f"split_seed{seed}.json"
    split_path.write_text(json.dumps({
        "seed": seed,
        "data_root": str(args.data_root),
        "classes": classes,
        "counts": {cls: {"train": sum(1 for _, l in train_items if l == i),
                          "test": sum(1 for _, l in test_items if l == i)}
                    for i, cls in enumerate(classes)},
        "train": [{"class": classes[l], "path": str(path.relative_to(args.data_root))}
                   for path, l in train_items],
        "test": [{"class": classes[l], "path": str(path.relative_to(args.data_root))}
                  for path, l in test_items],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    out_path = args.data_root / f"result_metrics_seed{seed}.json"
    out_path.write_text(json.dumps({"metrics": metrics, "classification_report": report},
                                    indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMetrics saved: {out_path}")
    print(f"Split saved:   {split_path}")


if __name__ == "__main__":
    main()
