#!/usr/bin/env python3
"""Score generated .ras candidates and build a visual contact sheet.

Use this in Colab after running invert_candidates.py. The scores tell us how
strongly the reconstructed classifier likes each candidate; the contact sheet
lets us judge whether direct optimization made recognizable images or just
classifier-triggering texture.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from preprocess_compare import preprocess
from reconstruct_model import build_resnet18, load_flat_weights, reconstruct_state_dict


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = build_resnet18()
    if args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
    else:
        flat = load_flat_weights(args.weights)
        state = reconstruct_state_dict(model, flat)
        model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def score_image(model: torch.nn.Module, path: Path, device: torch.device) -> tuple[float, float]:
    tensor = preprocess(path)
    x = torch.from_numpy(tensor.copy()).unsqueeze(0).to(device)
    with torch.inference_mode():
        logit = model(x).reshape(()).item()
        probability = torch.sigmoid(torch.tensor(logit)).item()
    return logit, probability


def make_contact_sheet(paths: list[Path], output: Path, tile_size: int = 160) -> None:
    tiles = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not read {path}")
        image = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        tiles.append(image)

    if not tiles:
        raise ValueError("no images to place in contact sheet")

    cols = min(6, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.full((rows * tile_size, cols * tile_size, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // cols
        col = index % cols
        y0 = row * tile_size
        x0 = col * tile_size
        sheet[y0 : y0 + tile_size, x0 : x0 + tile_size] = tile

    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise ValueError(f"OpenCV could not write {output}")


def resolve_candidate_paths(candidate_dir: Path, pattern: str | None) -> list[Path]:
    if pattern is not None:
        return sorted(candidate_dir.glob(pattern))
    paths = sorted(candidate_dir.glob("candidate_*.ras"))
    if paths:
        return paths
    return sorted(candidate_dir.glob("final_*.ras"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("analysis/candidates_first_pass"),
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Glob pattern. Defaults to candidate_*.ras, then final_*.ras if no candidates exist.",
    )
    parser.add_argument("--weights", type=Path, default=Path("analysis/resnet18_weights.bin"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("analysis/reconstructed_resnet18.pth"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("analysis/candidates_first_pass/scores.csv"),
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=Path("analysis/candidates_first_pass/contact_sheet.jpg"),
    )
    parser.add_argument(
        "--skip-contact-sheet",
        action="store_true",
        help="Write scores.csv only. Useful for large retrieval pools where one contact sheet would be too tall.",
    )
    parser.add_argument(
        "--contact-sheet-limit",
        type=int,
        default=None,
        help="Maximum number of images to include in the contact sheet.",
    )
    parser.add_argument("--expect-count", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_candidate_paths(args.candidate_dir, args.pattern)
    print(f"candidate files: {len(paths)}")
    if len(paths) == 0:
        pattern = args.pattern or "candidate_*.ras or final_*.ras"
        raise SystemExit(f"no candidates matched {args.candidate_dir} / {pattern}")
    if args.expect_count is not None and len(paths) != args.expect_count:
        raise SystemExit(f"expected {args.expect_count} files, matched {len(paths)}")

    device = torch.device(args.device)
    model = load_model(args, device)

    rows = []
    for path in paths:
        logit, probability = score_image(model, path, device)
        rows.append((path.name, logit, probability))

    rows.sort(key=lambda row: row[1], reverse=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "logit", "sigmoid"])
        writer.writerows(rows)

    if not args.skip_contact_sheet:
        sheet_paths = paths[: args.contact_sheet_limit] if args.contact_sheet_limit is not None else paths
        make_contact_sheet(sheet_paths, args.contact_sheet)

    print(f"wrote: {args.csv}")
    if not args.skip_contact_sheet:
        print(f"wrote: {args.contact_sheet}")
    print("top candidates:")
    for filename, logit, probability in rows[:10]:
        print(f"{filename} logit={logit:.6f} sigmoid={probability:.6f}")


if __name__ == "__main__":
    main()
