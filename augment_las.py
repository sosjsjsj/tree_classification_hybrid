"""
augment_las.py — Takes a single LAS file and writes 3 LAS files, each with a
different periphery augmentation applied.

How it works (duplicates the tree's peripheral points and pushes them outward):
  1. Compute each point's radius from the tree center (cx, cy)
  2. Split the z axis into thirds (lower/middle/upper) and take the top 20% by
     radius (the periphery) within each section
  3. Divide the peripheral points into 6 azimuth sectors of 60 degrees each
  4. Pick the 2 sectors matching the variant (0-2), including the opposite one,
     and duplicate their points
  5. Push the duplicates outward, away from the center, and add Gaussian noise
  6. Write the original point cloud plus the augmented points to a new LAS

Input:  1 LAS file
Output: <name>_aug1.las, <name>_aug2.las, <name>_aug3.las (variants 0/1/2)

Usage:
    python augment_las.py <input.las> [--seed 42] [--outdir DIR]
"""

from __future__ import annotations
import argparse
from pathlib import Path

import laspy
import numpy as np


def load_las(path: Path) -> np.ndarray:
    """LAS -> (N, 4) [x, y, z, intensity]"""
    las = laspy.read(path)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float32)
    intensity = np.asarray(las.intensity, dtype=np.float32).reshape(-1, 1)
    if len(xyz) == 0:
        raise ValueError(f"Empty LAS: {path}")
    return np.concatenate((xyz, intensity), 1)


def periphery_augment(points: np.ndarray, variant: int, seed: int) -> np.ndarray:
    """Duplicate some boundary sectors of the tree, push them outward, add noise."""
    xyz = points[:, :3]
    cx = (xyz[:, 0].min() + xyz[:, 0].max()) / 2
    cy = (xyz[:, 1].min() + xyz[:, 1].max()) / 2
    dx, dy = xyz[:, 0] - cx, xyz[:, 1] - cy
    radius = np.hypot(dx, dy)
    low, high = np.percentile(xyz[:, 2], (100 / 3, 200 / 3))
    sections = (xyz[:, 2] < low, (xyz[:, 2] >= low) & (xyz[:, 2] < high), xyz[:, 2] >= high)
    boundary = np.zeros(len(points), bool)
    for section in sections:
        if section.any():
            boundary |= section & (radius >= np.percentile(radius[section], 80))
    indices = np.flatnonzero(boundary)
    sectors = ((np.degrees(np.arctan2(dy[indices], dx[indices])) % 360) / 60).astype(int) % 6
    active = indices[np.isin(sectors, (variant, variant + 3))]
    repeated = np.tile(active, 2)
    duplicate = points[repeated].copy()
    norm = np.maximum(np.hypot(dx[repeated], dy[repeated]), 1e-12)
    rng = np.random.default_rng(seed)
    duplicate[:, 0] += 0.25 * dx[repeated] / norm
    duplicate[:, 1] += 0.25 * dy[repeated] / norm
    duplicate[:, :3] += rng.normal(0, 0.06, (len(duplicate), 3)).astype(np.float32)
    return np.concatenate((points, duplicate), 0)


def save_las(points: np.ndarray, template_path: Path, out_path: Path) -> None:
    """points (N,4)[x,y,z,intensity] -> saved reusing the template's header/format.
    Points added by augmentation get 0/default for the remaining attributes
    (classification and so on)."""
    template = laspy.read(template_path)
    n_orig = len(template.points)
    n_total = len(points)
    n_new = n_total - n_orig

    header = laspy.LasHeader(point_format=template.header.point_format,
                              version=template.header.version)
    header.scales = template.header.scales
    header.offsets = template.header.offsets

    out = laspy.LasData(header)
    out.x = points[:, 0]
    out.y = points[:, 1]
    out.z = points[:, 2]
    out.intensity = points[:, 3].astype(np.uint16)

    # Copy the remaining attributes (classification, return_number, ...) verbatim
    # for the original points; fill 0 (the default) for the newly augmented ones.
    for dim in template.point_format.dimension_names:
        if dim in ("X", "Y", "Z", "intensity"):
            continue
        try:
            orig_values = np.asarray(getattr(template, dim))
            new_values = np.zeros(n_total, dtype=orig_values.dtype)
            new_values[:n_orig] = orig_values
            setattr(out, dim, new_values)
        except Exception:
            pass  # skip read-only/derived attributes

    out.write(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="path to the input LAS file")
    parser.add_argument("--seed", type=int, default=42, help="seed for augmentation noise (default 42)")
    parser.add_argument("--outdir", type=Path, default=None, help="output directory (default: alongside the input file)")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    outdir = args.outdir or args.input.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    points = load_las(args.input)
    print(f"input: {args.input}  (points={len(points)})")

    for variant in range(3):
        seed = args.seed + variant
        augmented = periphery_augment(points, variant, seed)
        out_path = outdir / f"{stem}_aug{variant + 1}.las"
        save_las(augmented, args.input, out_path)
        print(f"  variant {variant}  ->  {out_path}  (points={len(augmented)})")

    print("done: 3 files written")


if __name__ == "__main__":
    main()
