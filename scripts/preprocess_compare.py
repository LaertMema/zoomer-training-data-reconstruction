#!/usr/bin/env python3
"""Compare Python preprocessing against a tensor dumped from the classifier.

This checks the image transform before model inversion:

    OpenCV BGR load -> resize shorter side to 256 -> center crop 224
    -> BGR to RGB -> ImageNet normalization -> NCHW float32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from image_pipeline import preprocess_path


def preprocess(path: Path) -> np.ndarray:
    return preprocess_path(path)


def compare(candidate: np.ndarray, target_path: Path) -> None:
    target = np.fromfile(target_path, dtype="<f4")
    if target.size != candidate.size:
        raise ValueError(
            f"{target_path}: expected {candidate.size} floats, got {target.size}"
        )
    target = target.reshape(candidate.shape)
    diff = candidate - target

    print(f"target tensor: {target_path}")
    print(f"shape: {candidate.shape}")
    print(f"mae: {np.mean(np.abs(diff)):.9f}")
    print(f"rmse: {np.sqrt(np.mean(diff * diff)):.9f}")
    print(f"max_abs: {np.max(np.abs(diff)):.9f}")
    print(f"candidate min/max: {candidate.min():.9f} {candidate.max():.9f}")
    print(f"target min/max: {target.min():.9f} {target.max():.9f}")

    for channel in range(3):
        channel_diff = diff[channel]
        print(
            f"channel {channel}: "
            f"mae={np.mean(np.abs(channel_diff)):.9f} "
            f"max_abs={np.max(np.abs(channel_diff)):.9f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=Path("executables/sample.ras"))
    parser.add_argument("--target", type=Path, default=Path("analysis/sample_tensor.bin"))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the Python-produced tensor.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tensor = preprocess(args.image)
    compare(tensor, args.target)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tensor.tofile(args.out)
        print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
