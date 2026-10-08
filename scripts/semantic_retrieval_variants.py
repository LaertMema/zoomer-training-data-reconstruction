#!/usr/bin/env python3
"""Build retrieval variants with external semantic embeddings.

This script complements select_retrieval_variants.py. The existing selector
uses reconstructed ResNet avgpool features; this one can use CLIP or DINOv2
features as a local proxy for the hidden semantic grader.

Expected input:

    pool_dir/
      retrieval_00000.ras
      retrieval_00001.ras
      scores.csv  # filename,logit,source_path

Typical Colab usage:

    python3 scripts/semantic_retrieval_variants.py \
      --pool-dir outputs/retrieval_pool \
      --encoder clip \
      --output-root outputs/semantic_variants_clip \
      --max-pool 500
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


def default_device() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def read_scores(csv_path: Path, pool_dir: Path) -> tuple[list[Path], np.ndarray, list[dict[str, str]]]:
    with csv_path.open() as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit(f"{csv_path} is empty")

    paths = []
    logits = []
    metadata = []
    for row in rows:
        path = pool_dir / row["filename"]
        paths.append(path)
        logits.append(float(row["logit"]))
        metadata.append(dict(row))
    return paths, np.asarray(logits, dtype=np.float32), metadata


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return (features / np.maximum(norm, 1e-12)).astype(np.float32)


def featurize_resnet(paths: list[Path], args: argparse.Namespace) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    from feature_select_candidates import FeatureHook, load_model
    from preprocess_compare import preprocess

    device = torch.device(args.device)
    model = load_model(args, device)
    hook = FeatureHook(model.avgpool)
    features = []
    try:
        for path in paths:
            tensor = preprocess(path)
            x = torch.from_numpy(tensor.copy()).unsqueeze(0).to(device)
            with torch.inference_mode():
                model(x)
                if hook.value is None:
                    raise RuntimeError("feature hook did not capture avgpool output")
                feature = F.normalize(hook.value, dim=1).squeeze(0).cpu().numpy()
            features.append(feature)
    finally:
        hook.close()
    return np.vstack(features).astype(np.float32)


def featurize_transformers(paths: list[Path], args: argparse.Namespace) -> np.ndarray:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel

    device = torch.device(args.device)
    if args.encoder == "clip":
        processor = AutoProcessor.from_pretrained(args.model)
        model = CLIPModel.from_pretrained(args.model).eval().to(device)
    elif args.encoder == "dino":
        processor = AutoImageProcessor.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model).eval().to(device)
    else:
        raise ValueError(f"unsupported transformers encoder: {args.encoder}")

    all_features = []
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        images = [Image.fromarray(read_rgb(path)) for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            if args.encoder == "clip":
                output = model.get_image_features(**inputs)
            else:
                output = model(**inputs)
                if hasattr(output, "pooler_output") and output.pooler_output is not None:
                    output = output.pooler_output
                else:
                    output = output.last_hidden_state[:, 0]
            if not isinstance(output, torch.Tensor):
                if hasattr(output, "image_embeds") and output.image_embeds is not None:
                    output = output.image_embeds
                elif hasattr(output, "pooler_output") and output.pooler_output is not None:
                    output = output.pooler_output
                elif hasattr(output, "last_hidden_state"):
                    output = output.last_hidden_state[:, 0]
                else:
                    raise TypeError(f"model returned unsupported output type: {type(output)}")
            output = torch.nn.functional.normalize(output, dim=1)
        all_features.append(output.cpu().numpy())
        print(f"embedded {min(start + len(batch_paths), len(paths))}/{len(paths)}")

    return np.vstack(all_features).astype(np.float32)


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


def select_top(logits: np.ndarray, count: int) -> list[int]:
    return np.argsort(-logits)[:count].astype(int).tolist()


def select_diverse(logits: np.ndarray, features: np.ndarray, count: int, score_weight: float) -> list[int]:
    selected = [int(np.argmax(logits))]
    while len(selected) < min(count, len(logits)):
        selected_features = features[selected]
        ranked = []
        for idx in range(len(logits)):
            if idx in selected:
                continue
            max_similarity = float((features[idx] @ selected_features.T).max())
            ranked.append((max_similarity - score_weight * float(logits[idx]), idx))
        selected.append(min(ranked)[1])
    return selected


def kmeans(features: np.ndarray, count: int, iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(features) < count:
        raise SystemExit(f"need at least {count} pool candidates, got {len(features)}")
    centers = features[rng.choice(len(features), size=count, replace=False)].copy()
    labels = np.zeros(len(features), dtype=np.int64)
    for _ in range(iterations):
        labels = np.argmax(features @ centers.T, axis=1)
        for cluster in range(count):
            members = features[labels == cluster]
            if len(members) == 0:
                centers[cluster] = features[rng.integers(len(features))]
            else:
                center = members.mean(axis=0)
                centers[cluster] = center / max(np.linalg.norm(center), 1e-12)
    return labels


def select_clusters(
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
        if members.size == 0:
            continue
        selected.append(int(members[np.argmax(logits[members])]))
    if len(selected) < count:
        for idx in np.argsort(-logits).astype(int).tolist():
            if idx not in selected:
                selected.append(idx)
            if len(selected) == count:
                break
    return selected[:count]


def pairwise_stats(features: np.ndarray, selected: list[int]) -> tuple[float, float]:
    selected_features = features[selected]
    similarity = selected_features @ selected_features.T
    if len(selected) <= 1:
        return 0.0, 0.0
    mask = ~np.eye(len(selected), dtype=bool)
    pairwise = similarity[mask]
    return float(pairwise.mean()), float(pairwise.max())


def write_variant(
    name: str,
    selected: list[int],
    paths: list[Path],
    logits: np.ndarray,
    metadata: list[dict[str, str]],
    features: np.ndarray,
    output_root: Path,
) -> dict[str, float | int | str]:
    if not selected:
        raise SystemExit(f"{name}: no candidates selected")

    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_features = features[selected]
    similarity = selected_features @ selected_features.T
    output_paths = []
    rows = []
    for rank, idx in enumerate(selected):
        destination = output_dir / f"candidate_{rank:02d}.ras"
        shutil.copy2(paths[idx], destination)
        output_paths.append(destination)
        others = [j for j in range(len(selected)) if j != rank]
        max_similarity = float(similarity[rank, others].max()) if others else 0.0
        row = {
            "rank": str(rank),
            "filename": destination.name,
            "pool_filename": paths[idx].name,
            "logit": f"{float(logits[idx]):.9f}",
            "max_selected_cosine": f"{max_similarity:.9f}",
        }
        for key, value in metadata[idx].items():
            if key in {"filename", "logit"}:
                continue
            row[key] = value
        rows.append(row)

    with (output_dir / "selection.csv").open("w", newline="") as file:
        fieldnames = ["rank", "filename", "pool_filename", "logit", "max_selected_cosine"]
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    make_contact_sheet(output_paths, output_dir / "contact_sheet.jpg")

    pairwise_mean, pairwise_max = pairwise_stats(features, selected)
    print(
        f"{name}: wrote {len(selected)} candidates to {output_dir} "
        f"logit_min={float(logits[selected].min()):.6f} "
        f"logit_mean={float(logits[selected].mean()):.6f} "
        f"pairwise_mean={pairwise_mean:.6f} "
        f"pairwise_max={pairwise_max:.6f}"
    )
    return {
        "variant": name,
        "output_dir": str(output_dir),
        "count": len(selected),
        "logit_min": float(logits[selected].min()),
        "logit_mean": float(logits[selected].mean()),
        "logit_max": float(logits[selected].max()),
        "pairwise_cos_mean": pairwise_mean,
        "pairwise_cos_max": pairwise_max,
    }


def write_summary(rows: list[dict[str, float | int | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "output_dir",
        "count",
        "logit_min",
        "logit_mean",
        "logit_max",
        "pairwise_cos_mean",
        "pairwise_cos_max",
    ]
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--scores-csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("analysis/semantic_variants"))
    parser.add_argument("--encoder", choices=["resnet", "clip", "dino"], default="clip")
    parser.add_argument("--model", default=None)
    parser.add_argument("--count", type=int, default=18)
    parser.add_argument("--max-pool", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cluster-seed", type=int, default=123)
    parser.add_argument("--cluster-iterations", type=int, default=25)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[
            "top_logits",
            "diverse",
            "diverse_logit_005",
            "diverse_logit_010",
            "cluster_representatives",
        ],
        default=None,
        help="Subset of variants to write. Defaults to all variants.",
    )
    parser.add_argument("--embedding-npz", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=Path("analysis/resnet18_weights.bin"))
    parser.add_argument("--checkpoint", type=Path, default=Path("analysis/reconstructed_resnet18.pth"))
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model is None:
        if args.encoder == "clip":
            args.model = "openai/clip-vit-base-patch32"
        elif args.encoder == "dino":
            args.model = "facebook/dinov2-base"
        else:
            args.model = "reconstructed_resnet18_avgpool"

    scores_csv = args.scores_csv or (args.pool_dir / "scores.csv")
    paths, logits, metadata = read_scores(scores_csv, args.pool_dir)
    if args.max_pool > 0:
        paths = paths[: args.max_pool]
        logits = logits[: args.max_pool]
        metadata = metadata[: args.max_pool]
    if len(paths) < args.count:
        raise SystemExit(f"need at least {args.count} pool candidates, got {len(paths)}")

    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing pool file: {missing[0]}")

    if args.embedding_npz is not None and args.embedding_npz.exists():
        data = np.load(args.embedding_npz, allow_pickle=True)
        features = data["features"].astype(np.float32)
        if features.shape[0] > len(paths):
            features = features[: len(paths)]
        print(f"loaded embeddings: {args.embedding_npz}")
    else:
        if args.encoder == "resnet":
            features = featurize_resnet(paths, args)
        else:
            features = featurize_transformers(paths, args)
        features = l2_normalize(features)
        if args.embedding_npz is not None:
            args.embedding_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.embedding_npz,
                features=features,
                filenames=np.asarray([path.name for path in paths]),
                source_paths=np.asarray([row.get("source_path", "") for row in metadata]),
                image_ids=np.asarray([row.get("image_id", "") for row in metadata]),
                categories=np.asarray([row.get("categories", "") for row in metadata]),
                logits=logits,
                encoder=args.encoder,
                model=args.model,
            )
            print(f"wrote embeddings: {args.embedding_npz}")

    if features.shape[0] != len(paths):
        raise SystemExit(
            f"embedding count {features.shape[0]} does not match path count {len(paths)}; "
            f"delete {args.embedding_npz} and rerun if the pool changed"
        )

    prefix = args.encoder
    all_variants = {
        "top_logits": select_top(logits, args.count),
        "diverse": select_diverse(logits, features, args.count, score_weight=0.0),
        "diverse_logit_005": select_diverse(logits, features, args.count, score_weight=0.05),
        "diverse_logit_010": select_diverse(logits, features, args.count, score_weight=0.10),
        "cluster_representatives": select_clusters(
            logits,
            features,
            args.count,
            args.cluster_iterations,
            args.cluster_seed,
        ),
    }
    selected_variant_names = args.variants or list(all_variants.keys())
    variants = {
        f"{prefix}_{name}": all_variants[name]
        for name in selected_variant_names
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, selected in variants.items():
        if len(selected) != args.count:
            raise SystemExit(f"{name}: expected {args.count}, selected {len(selected)}")
        rows.append(write_variant(name, selected, paths, logits, metadata, features, args.output_root))

    summary = args.output_root / f"{prefix}_summary.csv"
    write_summary(rows, summary)
    print(f"wrote summary: {summary}")


if __name__ == "__main__":
    main()
