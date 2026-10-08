#!/usr/bin/env python3
"""Retrieve positive candidates from an ImageNet-style image folder.

This branch tests whether the recovered classifier accepts natural images. It
scores ordinary images with the reconstructed ResNet-18, saves the strongest
positive candidates as 256x256 Sun raster files, and greedily selects a diverse
18-image subset using avgpool features.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from image_pipeline import crop_256_bgr, read_ras_bgr, write_ras_bgr
from reconstruct_model import build_resnet18, load_flat_weights, reconstruct_state_dict
from score_candidates import make_contact_sheet


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm", ".ras"}


class FeatureHook:
    def __init__(self, module: torch.nn.Module) -> None:
        self.value: torch.Tensor | None = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        del module, inputs
        self.value = output.flatten(1)

    def close(self) -> None:
        self.handle.remove()


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


def iter_images(root: Path, limit: int | None) -> list[Path]:
    paths = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            paths.append(path)
            if limit is not None and len(paths) >= limit:
                break
    return paths


def tensor_from_crops(crops_bgr: list[np.ndarray], device: torch.device) -> torch.Tensor:
    arrays = []
    for crop in crops_bgr:
        center = crop[16:240, 16:240]
        rgb = cv2.cvtColor(center, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = rgb.transpose(2, 0, 1)
        arrays.append(chw)

    batch = torch.from_numpy(np.stack(arrays, axis=0)).to(device=device)
    mean = MEAN.to(device=device, dtype=batch.dtype)
    std = STD.to(device=device, dtype=batch.dtype)
    return (batch - mean) / std


def push_top(
    heap: list[tuple[float, int, Path, np.ndarray, np.ndarray]],
    top_k: int,
    logit: float,
    serial: int,
    path: Path,
    crop: np.ndarray,
    feature: np.ndarray,
) -> None:
    item = (logit, serial, path, crop.copy(), feature.copy())
    if len(heap) < top_k:
        heapq.heappush(heap, item)
    elif logit > heap[0][0]:
        heapq.heapreplace(heap, item)


def greedy_select(logits: np.ndarray, features: np.ndarray, count: int, min_logit: float) -> list[int]:
    eligible = np.flatnonzero(logits >= min_logit)
    if eligible.size == 0:
        eligible = np.arange(len(logits))

    selected = [int(eligible[np.argmax(logits[eligible])])]
    while len(selected) < min(count, len(logits)):
        remaining = [idx for idx in eligible.tolist() if idx not in selected]
        if not remaining:
            remaining = [idx for idx in range(len(logits)) if idx not in selected]
        selected_features = features[selected]
        ranked = []
        for idx in remaining:
            max_similarity = float((features[idx] @ selected_features.T).max())
            score_bonus = 0.05 * float(logits[idx])
            ranked.append((max_similarity - score_bonus, idx))
        selected.append(min(ranked)[1])
    return selected


def score_saved_ras(model: torch.nn.Module, path: Path, device: torch.device) -> float:
    crop = crop_256_bgr(path)
    if crop is None:
        raise ValueError(f"OpenCV could not reload {path}")
    x = tensor_from_crops([crop], device)
    with torch.inference_mode():
        return float(model(x).reshape(()).item())


def run_classifier(classifier: Path, path: Path, target_output: str) -> tuple[int | None, str, bool | None]:
    if not classifier.exists():
        return None, "", None
    result = subprocess.run(
        [str(classifier), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout.strip()
    return result.returncode, output, result.returncode == 0 and output == target_output


def image_shape(path: Path) -> tuple[int | None, int | None]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None, None
    height, width = image.shape[:2]
    return height, width


def write_reload_audit(
    model: torch.nn.Module,
    paths: list[Path],
    before_logits: list[float],
    source_paths: list[Path],
    device: torch.device,
    classifier: Path,
    target_output: str,
    csv_path: Path,
    max_logit_delta: float,
    expected_ras_size: int,
    strict: bool,
) -> None:
    rows = []
    max_delta = 0.0
    binary_matches = 0
    binary_checked = 0
    failures = []
    if strict and not classifier.exists():
        failures.append(f"classifier not found: {classifier}")
    for path, before_logit, source_path in zip(paths, before_logits, source_paths):
        candidate_id = path.stem
        source_height, source_width = image_shape(source_path)
        try:
            ras_image = read_ras_bgr(path)
            reload_success = True
        except ValueError as error:
            ras_image = None
            reload_success = False
            failures.append(f"{path.name}: {error}")

        if ras_image is None:
            ras_height = None
            ras_width = None
            mean_b = mean_g = mean_r = float("nan")
            after_logit = float("nan")
            delta = float("inf")
            exit_code, output, target_match = None, "", False
        else:
            ras_height, ras_width = ras_image.shape[:2]
            if ras_height != expected_ras_size or ras_width != expected_ras_size:
                failures.append(
                    f"{path.name}: expected {expected_ras_size}x{expected_ras_size}, "
                    f"got {ras_width}x{ras_height}"
                )
            mean_b, mean_g, mean_r = ras_image.reshape(-1, 3).mean(axis=0)
            after_logit = score_saved_ras(model, path, device)
            delta = abs(before_logit - after_logit)
            if delta > max_logit_delta:
                failures.append(
                    f"{path.name}: reload logit delta {delta:.9f} exceeds {max_logit_delta:.9f}"
                )
            if strict:
                exit_code, output, target_match = run_classifier(classifier, path, target_output)
                if target_match is False:
                    failures.append(
                        f"{path.name}: binary output {output!r} exit={exit_code} "
                        f"does not match target {target_output!r}"
                    )
            else:
                exit_code, output, target_match = None, "", None

        delta = abs(before_logit - after_logit)
        max_delta = max(max_delta, delta)
        if target_match is not None:
            binary_checked += 1
            binary_matches += int(target_match)
        rows.append(
            (
                candidate_id,
                str(source_path),
                str(path),
                "" if source_height is None else source_height,
                "" if source_width is None else source_width,
                "" if ras_height is None else ras_height,
                "" if ras_width is None else ras_width,
                f"{before_logit:.9f}",
                f"{after_logit:.9f}",
                f"{delta:.9f}",
                output,
                "" if target_match is None else target_match,
                reload_success,
                f"{mean_b:.6f}",
                f"{mean_g:.6f}",
                f"{mean_r:.6f}",
                path.name,
                "" if exit_code is None else exit_code,
            )
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "candidate_id",
                "source_path",
                "ras_path",
                "source_height",
                "source_width",
                "ras_height",
                "ras_width",
                "logit_before_save",
                "logit_after_ras_reload",
                "logit_delta",
                "binary_output",
                "binary_target_match",
                "reload_success",
                "mean_b",
                "mean_g",
                "mean_r",
                "filename",
                "binary_exit_code",
            ]
        )
        writer.writerows(rows)

    print(f"wrote reload audit: {csv_path}")
    print(f"max reload logit delta: {max_delta:.9f}")
    if binary_checked:
        print(f"binary target matches: {binary_matches}/{binary_checked}")
    else:
        print(f"binary validation skipped; classifier not found: {classifier}")
    if failures:
        message = "reload audit failed:\n" + "\n".join(f"  {failure}" for failure in failures)
        if strict:
            raise SystemExit(message)
        print(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--select-count", type=int, default=18)
    parser.add_argument("--min-logit", type=float, default=0.0)
    parser.add_argument("--weights", type=Path, default=Path("analysis/resnet18_weights.bin"))
    parser.add_argument("--checkpoint", type=Path, default=Path("analysis/reconstructed_resnet18.pth"))
    parser.add_argument("--pool-dir", type=Path, default=Path("analysis/retrieval_pool"))
    parser.add_argument("--selected-dir", type=Path, default=Path("analysis/retrieval_selected_18"))
    parser.add_argument("--classifier", type=Path, default=Path("executables/linux-x86_64/classifier"))
    parser.add_argument("--target-output", default="0")
    parser.add_argument("--expect-count", type=int, default=18)
    parser.add_argument("--max-logit-delta", type=float, default=0.05)
    parser.add_argument("--expected-ras-size", type=int, default=256)
    parser.add_argument("--no-strict-audit", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args, device)
    hook = FeatureHook(model.avgpool)
    paths = iter_images(args.image_root, args.max_images)
    print(f"image files discovered: {len(paths)}")

    heap: list[tuple[float, int, Path, np.ndarray, np.ndarray]] = []
    accepted_seen = 0
    readable_seen = 0
    serial = 0

    try:
        for start in range(0, len(paths), args.batch_size):
            batch_paths = paths[start : start + args.batch_size]
            valid_paths = []
            crops = []
            for path in batch_paths:
                crop = crop_256_bgr(path)
                if crop is None:
                    continue
                valid_paths.append(path)
                crops.append(crop)

            if not crops:
                continue

            readable_seen += len(crops)
            x = tensor_from_crops(crops, device)
            with torch.inference_mode():
                logits = model(x).flatten()
                if hook.value is None:
                    raise RuntimeError("feature hook did not capture avgpool output")
                features = F.normalize(hook.value, dim=1).cpu().numpy()

            logits_cpu = logits.detach().cpu().numpy()
            for path, crop, feature, logit in zip(valid_paths, crops, features, logits_cpu):
                logit_value = float(logit)
                if logit_value >= args.min_logit:
                    accepted_seen += 1
                push_top(heap, args.top_k, logit_value, serial, path, crop, feature)
                serial += 1

            processed = min(start + args.batch_size, len(paths))
            if processed == len(paths) or processed % max(args.batch_size * 10, 1) == 0:
                best = max((item[0] for item in heap), default=float("nan"))
                print(
                    f"processed={processed} readable={readable_seen} "
                    f"positive_seen={accepted_seen} best_logit={best:.6f}"
                )
    finally:
        hook.close()

    items = sorted(heap, key=lambda item: item[0], reverse=True)
    if not items:
        raise SystemExit("no readable candidate images found")

    args.pool_dir.mkdir(parents=True, exist_ok=True)
    logits_np = np.asarray([item[0] for item in items], dtype=np.float32)
    features_np = np.vstack([item[4] for item in items]).astype(np.float32)

    pool_rows = []
    pool_paths = []
    for rank, (logit, _serial, source, crop, _feature) in enumerate(items):
        destination = args.pool_dir / f"retrieval_{rank:05d}.ras"
        write_ras_bgr(destination, crop)
        pool_paths.append(destination)
        pool_rows.append((destination.name, f"{logit:.9f}", str(source)))

    with (args.pool_dir / "scores.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "logit", "source_path"])
        writer.writerows(pool_rows)

    make_contact_sheet(pool_paths[: min(60, len(pool_paths))], args.pool_dir / "contact_sheet.jpg")

    selected = greedy_select(logits_np, features_np, args.select_count, args.min_logit)
    if args.expect_count is not None and len(selected) != args.expect_count:
        raise SystemExit(f"expected {args.expect_count} selected files, got {len(selected)}")
    args.selected_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    selected_paths = []
    selected_logits = []
    selected_sources = []
    similarity = features_np @ features_np.T
    for rank, idx in enumerate(selected):
        destination = args.selected_dir / f"candidate_{rank:02d}.ras"
        write_ras_bgr(destination, items[idx][3])
        selected_paths.append(destination)
        selected_logits.append(float(items[idx][0]))
        selected_sources.append(items[idx][2])
        selected_others = [j for j in selected if j != idx]
        max_selected_similarity = (
            float(similarity[idx, selected_others].max()) if selected_others else 0.0
        )
        selected_rows.append(
            (
                rank,
                destination.name,
                f"{items[idx][0]:.9f}",
                f"{max_selected_similarity:.9f}",
                str(items[idx][2]),
            )
        )
        print(
            f"{rank:02d} {destination.name} source={items[idx][2]} "
            f"logit={items[idx][0]:.6f} "
            f"max_selected_cos={max_selected_similarity:.6f}"
        )

    with (args.selected_dir / "selection.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["rank", "filename", "logit", "max_selected_cosine", "source_path"])
        writer.writerows(selected_rows)

    make_contact_sheet(selected_paths, args.selected_dir / "contact_sheet.jpg")
    write_reload_audit(
        model,
        selected_paths,
        selected_logits,
        selected_sources,
        device,
        args.classifier,
        args.target_output,
        args.selected_dir / "reload_audit.csv",
        args.max_logit_delta,
        args.expected_ras_size,
        not args.no_strict_audit,
    )
    print(f"wrote pool: {args.pool_dir}")
    print(f"wrote selected: {args.selected_dir}")
    print(f"positive seen at min_logit {args.min_logit}: {accepted_seen}/{readable_seen}")


if __name__ == "__main__":
    main()
