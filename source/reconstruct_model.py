#!/usr/bin/env python3
"""Reconstruct the embedded Zoomer ResNet-18 model from the dumped float blob.

Run from the repository root after installing torch and torchvision:

    python3 analysis/reconstruct_model.py

The script intentionally prints checkpoints for the core reverse-engineering
assumptions: blob size, state-dict loading, and logits for dumped binary input
tensors when they are available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


EXPECTED_FLOATS = 11_186_625
INPUT_FLOATS = 3 * 224 * 224


EXPECTED_MANIFEST = [
    ("conv1.weight", 0, 9408),
    ("bn1.weight", 9408, 64),
    ("bn1.bias", 9472, 64),
    ("bn1.running_mean", 9536, 64),
    ("bn1.running_var", 9600, 64),
    ("layer1.0.conv1.weight", 9664, 36864),
    ("layer1.0.bn1.weight", 46528, 64),
    ("layer1.0.bn1.bias", 46592, 64),
    ("layer1.0.bn1.running_mean", 46656, 64),
    ("layer1.0.bn1.running_var", 46720, 64),
    ("layer1.0.conv2.weight", 46784, 36864),
    ("layer1.0.bn2.weight", 83648, 64),
    ("layer1.0.bn2.bias", 83712, 64),
    ("layer1.0.bn2.running_mean", 83776, 64),
    ("layer1.0.bn2.running_var", 83840, 64),
    ("layer1.1.conv1.weight", 83904, 36864),
    ("layer1.1.bn1.weight", 120768, 64),
    ("layer1.1.bn1.bias", 120832, 64),
    ("layer1.1.bn1.running_mean", 120896, 64),
    ("layer1.1.bn1.running_var", 120960, 64),
    ("layer1.1.conv2.weight", 121024, 36864),
    ("layer1.1.bn2.weight", 157888, 64),
    ("layer1.1.bn2.bias", 157952, 64),
    ("layer1.1.bn2.running_mean", 158016, 64),
    ("layer1.1.bn2.running_var", 158080, 64),
    ("layer2.0.conv1.weight", 158144, 73728),
    ("layer2.0.bn1.weight", 231872, 128),
    ("layer2.0.bn1.bias", 232000, 128),
    ("layer2.0.bn1.running_mean", 232128, 128),
    ("layer2.0.bn1.running_var", 232256, 128),
    ("layer2.0.conv2.weight", 232384, 147456),
    ("layer2.0.bn2.weight", 379840, 128),
    ("layer2.0.bn2.bias", 379968, 128),
    ("layer2.0.bn2.running_mean", 380096, 128),
    ("layer2.0.bn2.running_var", 380224, 128),
    ("layer2.0.downsample.0.weight", 380352, 8192),
    ("layer2.0.downsample.1.weight", 388544, 128),
    ("layer2.0.downsample.1.bias", 388672, 128),
    ("layer2.0.downsample.1.running_mean", 388800, 128),
    ("layer2.0.downsample.1.running_var", 388928, 128),
    ("layer2.1.conv1.weight", 389056, 147456),
    ("layer2.1.bn1.weight", 536512, 128),
    ("layer2.1.bn1.bias", 536640, 128),
    ("layer2.1.bn1.running_mean", 536768, 128),
    ("layer2.1.bn1.running_var", 536896, 128),
    ("layer2.1.conv2.weight", 537024, 147456),
    ("layer2.1.bn2.weight", 684480, 128),
    ("layer2.1.bn2.bias", 684608, 128),
    ("layer2.1.bn2.running_mean", 684736, 128),
    ("layer2.1.bn2.running_var", 684864, 128),
    ("layer3.0.conv1.weight", 684992, 294912),
    ("layer3.0.bn1.weight", 979904, 256),
    ("layer3.0.bn1.bias", 980160, 256),
    ("layer3.0.bn1.running_mean", 980416, 256),
    ("layer3.0.bn1.running_var", 980672, 256),
    ("layer3.0.conv2.weight", 980928, 589824),
    ("layer3.0.bn2.weight", 1570752, 256),
    ("layer3.0.bn2.bias", 1571008, 256),
    ("layer3.0.bn2.running_mean", 1571264, 256),
    ("layer3.0.bn2.running_var", 1571520, 256),
    ("layer3.0.downsample.0.weight", 1571776, 32768),
    ("layer3.0.downsample.1.weight", 1604544, 256),
    ("layer3.0.downsample.1.bias", 1604800, 256),
    ("layer3.0.downsample.1.running_mean", 1605056, 256),
    ("layer3.0.downsample.1.running_var", 1605312, 256),
    ("layer3.1.conv1.weight", 1605568, 589824),
    ("layer3.1.bn1.weight", 2195392, 256),
    ("layer3.1.bn1.bias", 2195648, 256),
    ("layer3.1.bn1.running_mean", 2195904, 256),
    ("layer3.1.bn1.running_var", 2196160, 256),
    ("layer3.1.conv2.weight", 2196416, 589824),
    ("layer3.1.bn2.weight", 2786240, 256),
    ("layer3.1.bn2.bias", 2786496, 256),
    ("layer3.1.bn2.running_mean", 2786752, 256),
    ("layer3.1.bn2.running_var", 2787008, 256),
    ("layer4.0.conv1.weight", 2787264, 1179648),
    ("layer4.0.bn1.weight", 3966912, 512),
    ("layer4.0.bn1.bias", 3967424, 512),
    ("layer4.0.bn1.running_mean", 3967936, 512),
    ("layer4.0.bn1.running_var", 3968448, 512),
    ("layer4.0.conv2.weight", 3968960, 2359296),
    ("layer4.0.bn2.weight", 6328256, 512),
    ("layer4.0.bn2.bias", 6328768, 512),
    ("layer4.0.bn2.running_mean", 6329280, 512),
    ("layer4.0.bn2.running_var", 6329792, 512),
    ("layer4.0.downsample.0.weight", 6330304, 131072),
    ("layer4.0.downsample.1.weight", 6461376, 512),
    ("layer4.0.downsample.1.bias", 6461888, 512),
    ("layer4.0.downsample.1.running_mean", 6462400, 512),
    ("layer4.0.downsample.1.running_var", 6462912, 512),
    ("layer4.1.conv1.weight", 6463424, 2359296),
    ("layer4.1.bn1.weight", 8822720, 512),
    ("layer4.1.bn1.bias", 8823232, 512),
    ("layer4.1.bn1.running_mean", 8823744, 512),
    ("layer4.1.bn1.running_var", 8824256, 512),
    ("layer4.1.conv2.weight", 8824768, 2359296),
    ("layer4.1.bn2.weight", 11184064, 512),
    ("layer4.1.bn2.bias", 11184576, 512),
    ("layer4.1.bn2.running_mean", 11185088, 512),
    ("layer4.1.bn2.running_var", 11185600, 512),
    ("fc.weight", 11186112, 512),
    ("fc.bias", 11186624, 1),
]


def build_resnet18() -> nn.Module:
    try:
        from torchvision.models import resnet18
    except ImportError as exc:
        raise SystemExit(
            "torchvision is required. In Colab, run: "
            "pip install -q torch torchvision"
        ) from exc

    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 1)
    return model


def load_flat_weights(path: Path) -> np.ndarray:
    flat = np.fromfile(path, dtype="<f4")
    print(f"weights path: {path}")
    print(f"weights floats: {flat.size}")
    print(f"weights finite: {bool(np.isfinite(flat).all())}")
    if flat.size != EXPECTED_FLOATS:
        raise ValueError(f"expected {EXPECTED_FLOATS} floats, got {flat.size}")
    if not np.isfinite(flat).all():
        raise ValueError("weight blob contains non-finite values")
    return flat


def reconstruct_state_dict(model: nn.Module, flat: np.ndarray) -> dict[str, torch.Tensor]:
    templates = model.state_dict()
    manifest_by_name = {name: (offset, count) for name, offset, count in EXPECTED_MANIFEST}
    state = {}
    used = 0

    for name, template in templates.items():
        if name.endswith("num_batches_tracked"):
            state[name] = torch.zeros_like(template)
            continue

        if name not in manifest_by_name:
            raise KeyError(f"model state has unexpected parameter: {name}")

        offset, count = manifest_by_name[name]
        if count != template.numel():
            raise ValueError(
                f"{name}: manifest count {count} does not match model shape "
                f"{tuple(template.shape)} ({template.numel()} values)"
            )

        values = flat[offset : offset + count]
        if values.size != count:
            raise ValueError(f"{name}: blob ended early at offset {offset}")

        tensor = torch.from_numpy(values.reshape(tuple(template.shape)).copy())
        state[name] = tensor.to(dtype=template.dtype)
        used += count

    if used != EXPECTED_FLOATS:
        raise ValueError(f"used {used} floats, expected {EXPECTED_FLOATS}")

    return state


def load_tensor(path: Path, device: torch.device) -> torch.Tensor:
    values = np.fromfile(path, dtype="<f4")
    if values.size != INPUT_FLOATS:
        raise ValueError(f"{path}: expected {INPUT_FLOATS} floats, got {values.size}")
    x = torch.from_numpy(values.copy()).reshape(1, 3, 224, 224)
    return x.to(device=device)


def run_tensor_checks(model: nn.Module, tensor_paths: list[Path], device: torch.device) -> None:
    for path in tensor_paths:
        if not path.exists():
            print(f"skip missing tensor: {path}")
            continue
        x = load_tensor(path, device)
        with torch.inference_mode():
            logit = model(x).reshape(()).item()
            prob = torch.sigmoid(torch.tensor(logit)).item()
        print(f"{path}: logit={logit:.9f} sigmoid={prob:.9f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("analysis/resnet18_weights.bin"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("analysis/reconstructed_resnet18.pth"),
        help="Where to save the reconstructed PyTorch state_dict.",
    )
    parser.add_argument(
        "--tensor",
        action="append",
        type=Path,
        default=[
            Path("analysis/black_tensor.bin"),
            Path("analysis/white_tensor.bin"),
            Path("analysis/sample_tensor.bin"),
        ],
        help="Dumped NCHW float tensor to evaluate. May be repeated.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    flat = load_flat_weights(args.weights)
    model = build_resnet18()
    state = reconstruct_state_dict(model, flat)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"device: {device}")

    run_tensor_checks(model, args.tensor, device)


if __name__ == "__main__":
    main()
