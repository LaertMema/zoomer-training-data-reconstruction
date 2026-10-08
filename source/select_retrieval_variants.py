#!/usr/bin/env python3
"""Build multiple 18-image variants from a retrieval pool."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from preprocess_compare import preprocess


def read_scores(csv_path: Path, pool_dir: Path) -> tuple[list[Path], np.ndarray, list[str]]:
    with csv_path.open() as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit(f"{csv_path} is empty")

    paths = []
    logits = []
    sources = []
    for row in rows:
        path = pool_dir / row["filename"]
        paths.append(path)
        logits.append(float(row["logit"]))
        sources.append(row.get("source_path", ""))
    return paths, np.asarray(logits, dtype=np.float32), sources


def featurize_paths(
    model,
    paths: list[Path],
    device,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from feature_select_candidates import FeatureHook

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
                norm = np.linalg.norm(center)
                centers[cluster] = center / max(norm, 1e-12)
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


def write_variant(
    name: str,
    selected: list[int],
    paths: list[Path],
    logits: np.ndarray,
    sources: list[str],
    features: np.ndarray,
    output_root: Path,
) -> None:
    if not selected:
        raise SystemExit(f"{name}: no candidates selected")

    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_features = features[selected]
    similarity = selected_features @ selected_features.T
    rows = []
    output_paths = []
    for rank, idx in enumerate(selected):
        destination = output_dir / f"candidate_{rank:02d}.ras"
        shutil.copy2(paths[idx], destination)
        output_paths.append(destination)
        others = [j for j in range(len(selected)) if j != rank]
        max_similarity = float(similarity[rank, others].max()) if others else 0.0
        rows.append(
            (
                rank,
                destination.name,
                paths[idx].name,
                f"{float(logits[idx]):.9f}",
                f"{max_similarity:.9f}",
                sources[idx],
            )
        )

    with (output_dir / "selection.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["rank", "filename", "pool_filename", "logit", "max_selected_cosine", "source_path"]
        )
        writer.writerows(rows)
    make_contact_sheet(output_paths, output_dir / "contact_sheet.jpg")
    print(f"{name}: wrote {len(selected)} candidates to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--scores-csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("analysis/retrieval_variants"))
    parser.add_argument("--count", type=int, default=18)
    parser.add_argument("--max-pool", type=int, default=500)
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
    parser.add_argument("--weights", type=Path, default=Path("analysis/resnet18_weights.bin"))
    parser.add_argument("--checkpoint", type=Path, default=Path("analysis/reconstructed_resnet18.pth"))
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def default_device() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = parse_args()
    scores_csv = args.scores_csv or (args.pool_dir / "scores.csv")
    paths, logits, sources = read_scores(scores_csv, args.pool_dir)
    if args.max_pool > 0:
        paths = paths[: args.max_pool]
        logits = logits[: args.max_pool]
        sources = sources[: args.max_pool]
    if len(paths) < args.count:
        raise SystemExit(f"need at least {args.count} pool candidates, got {len(paths)}")

    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing pool file: {missing[0]}")

    import torch
    from feature_select_candidates import load_model

    device = torch.device(args.device)
    model = load_model(args, device)
    features = featurize_paths(model, paths, device)

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
    variant_names = args.variants or list(all_variants.keys())
    variants = {name: all_variants[name] for name in variant_names}

    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, selected in variants.items():
        if len(selected) != args.count:
            raise SystemExit(f"{name}: expected {args.count}, selected {len(selected)}")
        write_variant(name, selected, paths, logits, sources, features, args.output_root)


if __name__ == "__main__":
    main()
