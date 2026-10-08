#!/usr/bin/env python3
"""Small scene-prior retrieval branch for Zoomer backup exploration.

This scans a natural-image folder, uses CLIP text prompts to prefilter formal
indoor people scenes, scores the filtered images with the reconstructed
classifier, and writes only two challenge-sized variants: top18 and
CLIP-diverse.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import shutil
from pathlib import Path

import cv2
import numpy as np

from image_pipeline import crop_256_bgr, write_ras_bgr


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm", ".ras"}
MEAN_VALUES = [0.485, 0.456, 0.406]
STD_VALUES = [0.229, 0.224, 0.225]
DEFAULT_PROMPTS = [
    "formal people seated at a table with laptops",
    "business meeting with laptops and documents",
    "conference room with people using laptops",
    "classroom lecture room with laptops",
    "person in formal clothes working at a laptop",
    "office desk with person laptop chair documents",
    "presentation room with seated people and laptops",
    "people seated at table with notebooks and documents",
    "person using laptop at office desk",
    "seminar panel discussion with microphones and name tags",
    "person wearing tie using laptop in conference room",
    "whiteboard presentation with seated people and laptops",
    "computer keyboard on office desk with person seated",
    "formal shirt suit laptop chair office workstation",
]


class FeatureHook:
    def __init__(self, module: object) -> None:
        self.value = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module: object, inputs: tuple[object, ...], output: object) -> None:
        del module, inputs
        self.value = output.flatten(1)

    def close(self) -> None:
        self.handle.remove()


def make_contact_sheet(paths: list[Path], output: Path, tile_size: int = 160) -> None:
    tiles = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not read {path}")
        tiles.append(cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA))

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


def iter_images(root: Path, limit: int) -> list[Path]:
    paths = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            paths.append(path)
            if limit > 0 and len(paths) >= limit:
                break
    return paths


def default_device() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_classifier_model(args: argparse.Namespace, device: object) -> object:
    import torch

    from reconstruct_model import build_resnet18, load_flat_weights, reconstruct_state_dict

    model = build_resnet18()
    if args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
    else:
        flat = load_flat_weights(args.weights)
        state = reconstruct_state_dict(model, flat)
        model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def tensor_from_crops(crops_bgr: list[np.ndarray], device: object) -> object:
    import torch

    arrays = []
    for crop in crops_bgr:
        center = crop[16:240, 16:240]
        rgb = cv2.cvtColor(center, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        arrays.append(rgb.transpose(2, 0, 1))
    batch = torch.from_numpy(np.stack(arrays, axis=0)).to(device=device)
    mean = torch.tensor(MEAN_VALUES, dtype=batch.dtype, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD_VALUES, dtype=batch.dtype, device=device).view(1, 3, 1, 1)
    return (batch - mean) / std


def read_pil(path: Path) -> object | None:
    from PIL import Image

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    prompts = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not prompts:
        raise SystemExit(f"prompt file is empty: {path}")
    return prompts


def clip_output_tensor(output: object, kind: str) -> torch.Tensor:
    import torch

    if isinstance(output, torch.Tensor):
        return output
    attr_names = {
        "text": ("text_embeds", "pooler_output", "last_hidden_state"),
        "image": ("image_embeds", "pooler_output", "last_hidden_state"),
    }[kind]
    for attr_name in attr_names:
        value = getattr(output, attr_name, None)
        if value is None:
            continue
        if attr_name == "last_hidden_state":
            return value[:, 0]
        return value
    raise TypeError(f"CLIP {kind} encoder returned unsupported output type: {type(output)}")


def clip_prefilter(
    paths: list[Path],
    prompts: list[str],
    args: argparse.Namespace,
    device: object,
) -> tuple[list[Path], np.ndarray, np.ndarray, list[str]]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoProcessor, CLIPModel

    processor = AutoProcessor.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model).eval().to(device)

    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        text_features = clip_output_tensor(model.get_text_features(**text_inputs), "text")
        text_features = F.normalize(text_features, dim=1)

    heap: list[tuple[float, int, Path, np.ndarray, str]] = []
    readable = 0
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        images = []
        valid_paths = []
        for path in batch_paths:
            image = read_pil(path)
            if image is None:
                continue
            images.append(image)
            valid_paths.append(path)
        if not images:
            continue
        readable += len(images)
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            image_features = clip_output_tensor(model.get_image_features(**inputs), "image")
            image_features = F.normalize(image_features, dim=1)
            similarity = image_features @ text_features.T
        scores, prompt_indices = similarity.max(dim=1)
        features_cpu = image_features.cpu().numpy().astype(np.float32)
        for offset, (path, score, prompt_idx, feature) in enumerate(
            zip(valid_paths, scores.cpu().numpy(), prompt_indices.cpu().numpy(), features_cpu)
        ):
            item = (float(score), start + offset, path, feature.copy(), prompts[int(prompt_idx)])
            if len(heap) < args.clip_keep:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
        processed = min(start + args.batch_size, len(paths))
        if processed == len(paths) or processed % max(args.batch_size * 20, 1) == 0:
            best = max((item[0] for item in heap), default=float("nan"))
            print(f"clip processed={processed} readable={readable} kept={len(heap)} best_clip={best:.6f}")

    items = sorted(heap, key=lambda item: item[0], reverse=True)
    kept_paths = [item[2] for item in items]
    clip_scores = np.asarray([item[0] for item in items], dtype=np.float32)
    clip_features = np.vstack([item[3] for item in items]).astype(np.float32)
    best_prompts = [item[4] for item in items]
    print(f"CLIP-filtered images: {len(kept_paths)} from {len(paths)}")
    return kept_paths, clip_scores, clip_features, best_prompts


def score_classifier_pool(
    paths: list[Path],
    clip_scores: np.ndarray,
    clip_features: np.ndarray,
    best_prompts: list[str],
    args: argparse.Namespace,
    device: object,
) -> list[dict[str, object]]:
    import torch
    import torch.nn.functional as F

    model = load_classifier_model(args, device)
    hook = FeatureHook(model.avgpool)
    heap: list[tuple[float, int, dict[str, object]]] = []
    readable = 0
    try:
        for start in range(0, len(paths), args.classifier_batch_size):
            batch_paths = paths[start : start + args.classifier_batch_size]
            valid_offsets = []
            valid_paths = []
            crops = []
            for local_offset, path in enumerate(batch_paths):
                crop = crop_256_bgr(path)
                if crop is None:
                    continue
                valid_offsets.append(start + local_offset)
                valid_paths.append(path)
                crops.append(crop)
            if not crops:
                continue
            readable += len(crops)
            x = tensor_from_crops(crops, device)
            with torch.inference_mode():
                logits = model(x).flatten()
                if hook.value is None:
                    raise RuntimeError("feature hook did not capture avgpool output")
                resnet_features = F.normalize(hook.value, dim=1).cpu().numpy().astype(np.float32)
            for serial, path, crop, logit, feature in zip(
                valid_offsets,
                valid_paths,
                crops,
                logits.cpu().numpy(),
                resnet_features,
            ):
                row = {
                    "source_path": path,
                    "crop": crop.copy(),
                    "logit": float(logit),
                    "clip_score": float(clip_scores[serial]),
                    "clip_feature": clip_features[serial].copy(),
                    "resnet_feature": feature.copy(),
                    "best_prompt": best_prompts[serial],
                }
                item = (float(logit), serial, row)
                if len(heap) < args.classifier_keep:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)
            processed = min(start + args.classifier_batch_size, len(paths))
            if processed == len(paths) or processed % max(args.classifier_batch_size * 10, 1) == 0:
                best = max((item[0] for item in heap), default=float("nan"))
                print(f"classifier processed={processed} readable={readable} kept={len(heap)} best_logit={best:.6f}")
    finally:
        hook.close()

    rows = [item[2] for item in sorted(heap, key=lambda item: item[0], reverse=True)]
    print(f"classifier pool images: {len(rows)}")
    return rows


def select_clip_diverse(
    logits: np.ndarray,
    features: np.ndarray,
    count: int,
    score_weight: float = 0.0,
) -> list[int]:
    selected = [int(np.argmax(logits))]
    while len(selected) < min(count, len(logits)):
        selected_features = features[selected]
        candidates = []
        for idx in range(len(logits)):
            if idx in selected:
                continue
            max_similarity = float((features[idx] @ selected_features.T).max())
            candidates.append((max_similarity - score_weight * float(logits[idx]), idx))
        selected.append(min(candidates)[1])
    return selected


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return (features / np.maximum(norm, 1e-12)).astype(np.float32)


def kmeans(features: np.ndarray, count: int, iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(features) < count:
        raise SystemExit(f"need at least {count} features, got {len(features)}")
    centers = features[rng.choice(len(features), size=count, replace=False)].copy()
    labels = np.zeros(len(features), dtype=np.int64)
    for _ in range(iterations):
        labels = np.argmax(features @ centers.T, axis=1)
        for cluster in range(count):
            members = features[labels == cluster]
            if len(members) == 0:
                centers[cluster] = features[rng.integers(len(features))]
                continue
            center = members.mean(axis=0)
            centers[cluster] = center / max(float(np.linalg.norm(center)), 1e-12)
    return labels


def select_clip_clusters(
    logits: np.ndarray,
    features: np.ndarray,
    count: int,
    iterations: int,
    seed: int,
) -> list[int]:
    labels = kmeans(features, count, iterations, seed)
    selected = []
    for cluster in range(count):
        members = np.flatnonzero(labels == cluster)
        if members.size:
            selected.append(int(members[np.argmax(logits[members])]))
    if len(selected) < count:
        for idx in np.argsort(-logits).astype(int).tolist():
            if idx not in selected:
                selected.append(idx)
            if len(selected) == count:
                break
    return selected[:count]


def selected_indices_for_variant(
    variant: str,
    logits: np.ndarray,
    clip_features: np.ndarray,
    count: int,
    cluster_iterations: int,
    cluster_seed: int,
) -> list[int]:
    if variant == "top18":
        return np.argsort(-logits)[:count].astype(int).tolist()
    if variant == "clip_diverse":
        return select_clip_diverse(logits, clip_features, count, score_weight=0.0)
    if variant == "clip_diverse_logit_005":
        return select_clip_diverse(logits, clip_features, count, score_weight=0.05)
    if variant == "clip_diverse_logit_010":
        return select_clip_diverse(logits, clip_features, count, score_weight=0.10)
    if variant == "clip_cluster_representatives":
        return select_clip_clusters(logits, clip_features, count, cluster_iterations, cluster_seed)
    raise ValueError(f"unsupported variant: {variant}")


def write_pool(
    pool_dir: Path,
    rows: list[dict[str, object]],
) -> tuple[list[Path], np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]]]:
    pool_dir.mkdir(parents=True, exist_ok=True)
    pool_paths = []
    csv_rows = []
    for rank, row in enumerate(rows):
        destination = pool_dir / f"retrieval_{rank:05d}.ras"
        write_ras_bgr(destination, row["crop"])
        pool_paths.append(destination)
        csv_rows.append(
            {
                "filename": destination.name,
                "logit": f"{float(row['logit']):.9f}",
                "source_path": str(row["source_path"]),
                "clip_score": f"{float(row['clip_score']):.9f}",
                "best_prompt": str(row["best_prompt"]),
            }
        )
    with (pool_dir / "scores.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["filename", "logit", "source_path", "clip_score", "best_prompt"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    make_contact_sheet(pool_paths[: min(60, len(pool_paths))], pool_dir / "contact_sheet.jpg")
    logits = np.asarray([float(row["logit"]) for row in rows], dtype=np.float32)
    clip_features = np.vstack([row["clip_feature"] for row in rows]).astype(np.float32)
    clip_scores = np.asarray([float(row["clip_score"]) for row in rows], dtype=np.float32)
    return pool_paths, logits, clip_features, clip_scores, csv_rows


def write_variant(
    output_dir: Path,
    selected: list[int],
    pool_paths: list[Path],
    logits: np.ndarray,
    clip_features: np.ndarray,
    metadata: list[dict[str, str]],
    method: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_features = clip_features[selected]
    similarity = selected_features @ selected_features.T
    output_paths = []
    rows = []
    for rank, idx in enumerate(selected):
        destination = output_dir / f"candidate_{rank:02d}.ras"
        shutil.copy2(pool_paths[idx], destination)
        output_paths.append(destination)
        others = [j for j in range(len(selected)) if j != rank]
        max_similarity = float(similarity[rank, others].max()) if others else 0.0
        rows.append(
            {
                "rank": rank,
                "filename": destination.name,
                "pool_filename": pool_paths[idx].name,
                "logit": f"{float(logits[idx]):.9f}",
                "max_selected_clip_cosine": f"{max_similarity:.9f}",
                "source_path": metadata[idx].get("source_path", ""),
                "clip_score": metadata[idx].get("clip_score", ""),
                "best_prompt": metadata[idx].get("best_prompt", ""),
                "selection_method": method,
            }
        )
    with (output_dir / "selection.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    make_contact_sheet(output_paths, output_dir / "contact_sheet.jpg")
    print(f"{output_dir.name}: wrote {len(selected)} candidates")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("analysis/semantic_scene_retrieval"))
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--clip-keep", type=int, default=3000)
    parser.add_argument("--classifier-keep", type=int, default=750)
    parser.add_argument("--select-count", type=int, default=18)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[
            "top18",
            "clip_diverse",
            "clip_diverse_logit_005",
            "clip_diverse_logit_010",
            "clip_cluster_representatives",
        ],
        default=["top18", "clip_diverse", "clip_diverse_logit_005", "clip_cluster_representatives"],
    )
    parser.add_argument("--cluster-seed", type=int, default=123)
    parser.add_argument("--cluster-iterations", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--classifier-batch-size", type=int, default=128)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--weights", type=Path, default=Path("analysis/resnet18_weights.bin"))
    parser.add_argument("--checkpoint", type=Path, default=Path("analysis/reconstructed_resnet18.pth"))
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    device = torch.device(args.device)
    prompts = load_prompts(args.prompt_file)
    paths = iter_images(args.image_root, args.max_images)
    print(f"image files discovered: {len(paths)}")
    if len(paths) < args.select_count:
        raise SystemExit(f"need at least {args.select_count} images, found {len(paths)}")

    clip_paths, clip_scores, clip_features, best_prompts = clip_prefilter(paths, prompts, args, device)
    if len(clip_paths) < args.select_count:
        raise SystemExit(f"CLIP prefilter produced fewer than {args.select_count} images: {len(clip_paths)}")

    rows = score_classifier_pool(clip_paths, clip_scores, clip_features, best_prompts, args, device)
    if len(rows) < args.select_count:
        raise SystemExit(f"classifier pool produced fewer than {args.select_count} images: {len(rows)}")

    label_root = args.output_root / args.label
    pool_paths, logits, pool_clip_features, _pool_clip_scores, metadata = write_pool(label_root / "pool", rows)
    pool_clip_features = l2_normalize(pool_clip_features)
    for variant in args.variants:
        selected = selected_indices_for_variant(
            variant,
            logits,
            pool_clip_features,
            args.select_count,
            args.cluster_iterations,
            args.cluster_seed,
        )
        if len(selected) != args.select_count:
            raise SystemExit(f"{variant}: expected {args.select_count} selected images, got {len(selected)}")
        write_variant(
            label_root / f"{args.label}_{variant}",
            selected,
            pool_paths,
            logits,
            pool_clip_features,
            metadata,
            variant,
        )
    print(f"wrote scene-prior outputs to: {label_root}")


if __name__ == "__main__":
    main()
