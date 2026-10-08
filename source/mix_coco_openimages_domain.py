#!/usr/bin/env python3
"""Build fixed-domain COCO/Open Images mixed candidate sets.

The anchor domain is the current best hidden-score proxy:

    formal or semi-formal person + tie/formalwear + laptop + chair/seated
    office/classroom/conference/workstation context

This script does not mix datasets randomly. It treats the current best
formal-workstation set as the anchor, scores replacement candidates by CLIP
similarity to that anchor plus light classifier/theme pressure, and writes
controlled replacement ratios such as 15 anchor + 3 replacement and 12 + 6.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_COCO_ANCHOR = Path(
    "analysis/person_tie_office_refinement_candidate_subsets/analysis/"
    "person_tie_office_refinement/semantic_clip/person_tie_laptop_chair/"
    "clip_cluster_representatives"
)

DEFAULT_OPENIMAGES_ANCHOR = Path(
    "analysis/anchor_ablation_tests/ablate_anchor17_neutral_filler_035"
)

DEFAULT_ANCHOR = DEFAULT_OPENIMAGES_ANCHOR

DEFAULT_PROMPTS = [
    "a formal person seated at a laptop in an office",
    "a semi formal person with a laptop and chair at a workstation",
    "a person wearing a tie using a laptop at a desk",
    "people seated with laptops in a conference room",
    "a classroom or lecture room with seated people using laptops",
    "an office workstation with a person laptop chair and table",
    "a conference poster session with students using laptops",
    "a person using a laptop beside an academic research poster",
    "a classroom with posters and students using computers",
    "a computer science conference booth with a poster and laptop",
    "a technical workshop with laptops projector screen and poster boards",
    "a student conference presentation board with laptops",
    "a research demo table with laptops documents and poster board",
    "a formal academic conference attendee in a suit using a laptop",
    "a person wearing a tie at a conference poster with a laptop",
    "a suited presenter with a laptop near an academic poster board",
]


@dataclass
class Candidate:
    path: Path
    source: str
    source_variant: str
    logit: float
    metadata: dict[str, str]
    digest: str


def default_device() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


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


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return (features / np.maximum(norm, 1e-12)).astype(np.float32)


def read_selection_rows(directory: Path) -> dict[str, dict[str, str]]:
    path = directory / "selection.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as file:
        return {row["filename"]: dict(row) for row in csv.DictReader(file)}


def candidate_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    dirs = []
    for path in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        if len(list(path.glob("candidate_*.ras"))) == 18:
            dirs.append(path)
    return sorted(set(dirs))


def collect_candidates(
    root: Path,
    source: str,
    include_substrings: list[str],
    limit_per_dir: int,
) -> list[Candidate]:
    candidates = []
    seen: set[str] = set()
    dirs = candidate_dirs(root)
    if not dirs and len(list(root.glob("candidate_*.ras"))) == 18:
        dirs = [root]

    for directory in dirs:
        rel = directory.relative_to(root).as_posix() if directory != root else directory.name
        if include_substrings and not any(token in rel for token in include_substrings):
            continue
        rows = read_selection_rows(directory)
        paths = sorted(directory.glob("candidate_*.ras"))
        if limit_per_dir > 0:
            paths = paths[:limit_per_dir]
        for path in paths:
            digest = file_digest(path)
            if digest in seen:
                continue
            seen.add(digest)
            row = rows.get(path.name, {})
            try:
                logit = float(row.get("logit", "0"))
            except ValueError:
                logit = 0.0
            metadata = dict(row)
            metadata.setdefault("source_path", str(path))
            candidates.append(
                Candidate(
                    path=path,
                    source=source,
                    source_variant=rel,
                    logit=logit,
                    metadata=metadata,
                    digest=digest,
                )
            )
    return candidates


def featurize_clip(
    paths: list[Path],
    prompts: list[str],
    model_name: str,
    device_name: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, CLIPModel

    def clip_tensor(output: object, kind: str):
        if isinstance(output, torch.Tensor):
            return output
        if kind == "image" and hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds
        if kind == "text" and hasattr(output, "text_embeds") and output.text_embeds is not None:
            return output.text_embeds
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            return output.last_hidden_state[:, 0]
        raise TypeError(f"CLIP returned unsupported {kind} output type: {type(output)}")

    device = torch.device(device_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).eval().to(device)

    image_features = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = [Image.fromarray(read_rgb(path)) for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.get_image_features(**inputs)
            output = clip_tensor(output, "image")
            output = torch.nn.functional.normalize(output, dim=1)
        image_features.append(output.cpu().numpy())
        print(f"embedded images {min(start + len(batch_paths), len(paths))}/{len(paths)}")

    text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    with torch.inference_mode():
        text_features = model.get_text_features(**text_inputs)
        text_features = clip_tensor(text_features, "text")
        text_features = torch.nn.functional.normalize(text_features, dim=1)

    return (
        np.vstack(image_features).astype(np.float32),
        text_features.cpu().numpy().astype(np.float32),
    )


def normalized(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.astype(np.float32)
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def select_top(pool: list[int], score: np.ndarray, count: int) -> list[int]:
    ranked = sorted(pool, key=lambda idx: float(score[idx]), reverse=True)
    return ranked[:count]


def select_diverse(pool: list[int], score: np.ndarray, features: np.ndarray, count: int, score_weight: float) -> list[int]:
    if count == 0:
        return []
    selected = [max(pool, key=lambda idx: float(score[idx]))]
    remaining = set(pool) - set(selected)
    while len(selected) < count and remaining:
        selected_features = features[selected]
        best_idx = None
        best_value = None
        for idx in remaining:
            max_similarity = float((features[idx] @ selected_features.T).max())
            value = max_similarity - score_weight * float(score[idx])
            if best_value is None or value < best_value:
                best_value = value
                best_idx = idx
        selected.append(int(best_idx))
        remaining.remove(int(best_idx))
    return selected


def kmeans(features: np.ndarray, count: int, iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
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


def select_clusters(pool: list[int], score: np.ndarray, features: np.ndarray, count: int, seed: int) -> list[int]:
    if count == 0:
        return []
    if len(pool) <= count:
        return select_top(pool, score, count)
    pool_features = features[pool]
    labels = kmeans(pool_features, count, iterations=25, seed=seed)
    selected = []
    for cluster in range(count):
        member_positions = np.flatnonzero(labels == cluster)
        if member_positions.size == 0:
            continue
        members = [pool[int(position)] for position in member_positions]
        selected.append(max(members, key=lambda idx: float(score[idx])))
    if len(selected) < count:
        for idx in select_top(pool, score, len(pool)):
            if idx not in selected:
                selected.append(idx)
            if len(selected) == count:
                break
    return selected[:count]


def parse_ratios(raw: str) -> list[tuple[int, int]]:
    ratios = []
    for item in raw.split(","):
        left, right = item.strip().split(":")
        ratios.append((int(left), int(right)))
    return ratios


def parse_counts(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def select_variant(
    variant: str,
    anchor_pool: list[int],
    open_pool: list[int],
    score: np.ndarray,
    features: np.ndarray,
    coco_count: int,
    open_count: int,
    cluster_seed: int,
) -> list[int]:
    if variant == "top18":
        selected = select_top(anchor_pool, score, coco_count) + select_top(open_pool, score, open_count)
    elif variant == "diverse":
        selected = select_diverse(anchor_pool, score, features, coco_count, score_weight=0.0)
        selected += select_diverse(open_pool, score, features, open_count, score_weight=0.0)
    elif variant == "diverse_logit_005":
        selected = select_diverse(anchor_pool, score, features, coco_count, score_weight=0.05)
        selected += select_diverse(open_pool, score, features, open_count, score_weight=0.05)
    elif variant == "clustered":
        selected = select_clusters(anchor_pool, score, features, coco_count, seed=cluster_seed)
        selected += select_clusters(open_pool, score, features, open_count, seed=cluster_seed + 1)
    else:
        raise AssertionError(variant)
    return sorted(selected, key=lambda idx: float(score[idx]), reverse=True)


def write_variant(
    name: str,
    selected: list[int],
    candidates: list[Candidate],
    score: np.ndarray,
    anchor_sim: np.ndarray,
    theme_score: np.ndarray,
    output_root: Path,
) -> None:
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    rows = []
    for rank, idx in enumerate(selected):
        candidate = candidates[idx]
        destination = output_dir / f"candidate_{rank:02d}.ras"
        shutil.copy2(candidate.path, destination)
        output_paths.append(destination)
        row = {
            "rank": str(rank),
            "filename": destination.name,
            "source_dataset": candidate.source,
            "source_variant": candidate.source_variant,
            "source_file": str(candidate.path),
            "logit": f"{candidate.logit:.9f}",
            "anchor_clip_similarity": f"{float(anchor_sim[idx]):.9f}",
            "theme_clip_score": f"{float(theme_score[idx]):.9f}",
            "mixed_score": f"{float(score[idx]):.9f}",
            "sha256": candidate.digest,
        }
        for key, value in candidate.metadata.items():
            if key not in row and key not in {"rank", "filename"}:
                row[key] = value
        rows.append(row)

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "selection.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    make_contact_sheet(output_paths, output_dir / "contact_sheet.jpg")

    coco_count = sum(candidates[idx].source == "coco_anchor" for idx in selected)
    open_count = sum(candidates[idx].source == "openimages" for idx in selected)
    print(f"{name}: wrote {len(selected)} candidates ({coco_count} COCO, {open_count} Open Images) to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-dir", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument(
        "--openimages-root",
        type=Path,
        default=Path("analysis/openimages_domain_candidate_subsets/analysis/openimages_domain_retrieval"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("analysis/coco_openimages_domain_mix"))
    parser.add_argument(
        "--mode",
        choices=["fixed_ratios", "replace_weakest", "both"],
        default="fixed_ratios",
        help="fixed_ratios writes the coarse ratio grid; replace_weakest writes fine-grained 1-6 Open Images replacement sets.",
    )
    parser.add_argument("--ratios", default="18:0,15:3,12:6,9:9,6:12,0:18")
    parser.add_argument("--replacement-counts", default="1,2,3,4,5,6")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["clustered", "diverse_logit_005", "diverse", "top18"],
        default=["clustered", "diverse_logit_005", "diverse", "top18"],
    )
    parser.add_argument(
        "--openimages-include",
        nargs="*",
        default=["person_tie", "person_laptop", "person_desk", "person_book", "person_mobile"],
        help="Only use Open Images candidate directories whose relative path contains one of these tokens.",
    )
    parser.add_argument("--limit-per-openimages-dir", type=int, default=18)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--logit-weight", type=float, default=0.05)
    parser.add_argument("--theme-weight", type=float, default=0.25)
    parser.add_argument("--cluster-seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anchor = collect_candidates(args.anchor_dir, "coco_anchor", include_substrings=[], limit_per_dir=18)
    openimages = collect_candidates(
        args.openimages_root,
        "openimages",
        include_substrings=args.openimages_include,
        limit_per_dir=args.limit_per_openimages_dir,
    )
    if len(anchor) < 18:
        raise SystemExit(f"anchor must contain 18 candidates, got {len(anchor)} from {args.anchor_dir}")
    if not openimages:
        raise SystemExit(f"no Open Images candidates found under {args.openimages_root}")

    candidates = anchor + openimages
    paths = [candidate.path for candidate in candidates]
    print(f"candidate pool: {len(anchor)} COCO anchor + {len(openimages)} Open Images")
    features, text_features = featurize_clip(paths, DEFAULT_PROMPTS, args.model, args.device, args.batch_size)
    features = l2_normalize(features)

    anchor_features = features[: len(anchor)]
    anchor_center = anchor_features.mean(axis=0)
    anchor_center = anchor_center / max(float(np.linalg.norm(anchor_center)), 1e-12)
    anchor_sim = features @ anchor_center
    theme_score = (features @ text_features.T).max(axis=1)
    logits = np.asarray([candidate.logit for candidate in candidates], dtype=np.float32)
    score = anchor_sim + args.theme_weight * theme_score + args.logit_weight * normalized(logits)

    anchor_pool = list(range(len(anchor)))
    open_pool = list(range(len(anchor), len(candidates)))
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    jobs = []
    if args.mode in {"fixed_ratios", "both"}:
        for coco_count, open_count in parse_ratios(args.ratios):
            jobs.append(("fixed_ratio", f"mix_{coco_count:02d}coco_{open_count:02d}openimages", coco_count, open_count))
    if args.mode in {"replace_weakest", "both"}:
        for open_count in parse_counts(args.replacement_counts):
            coco_count = 18 - open_count
            jobs.append(("replace_weakest", f"replace_{open_count:02d}openimages", coco_count, open_count))

    for mode, prefix, coco_count, open_count in jobs:
        if coco_count + open_count != 18:
            raise SystemExit(f"ratio {coco_count}:{open_count} does not sum to 18")
        if coco_count > len(anchor):
            raise SystemExit(f"ratio needs {coco_count} COCO candidates, only {len(anchor)} available")
        if open_count > len(open_pool):
            raise SystemExit(f"ratio needs {open_count} Open Images candidates, only {len(open_pool)} available")

        for variant in args.variants:
            selected = select_variant(
                variant,
                anchor_pool,
                open_pool,
                score,
                features,
                coco_count,
                open_count,
                args.cluster_seed,
            )
            name = f"{prefix}_{variant}"
            write_variant(name, selected, candidates, score, anchor_sim, theme_score, args.output_root)
            summary_rows.append(
                {
                    "variant": name,
                    "mode": mode,
                    "coco_count": coco_count,
                    "openimages_count": open_count,
                    "score_mean": f"{float(score[selected].mean()):.9f}",
                    "anchor_similarity_mean": f"{float(anchor_sim[selected].mean()):.9f}",
                    "theme_score_mean": f"{float(theme_score[selected].mean()):.9f}",
                    "logit_mean": f"{float(logits[selected].mean()):.9f}",
                    "output_dir": str(args.output_root / name),
                }
            )

    summary = args.output_root / "mixed_summary.csv"
    with summary.open("w", newline="") as file:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote summary: {summary}")


if __name__ == "__main__":
    main()
