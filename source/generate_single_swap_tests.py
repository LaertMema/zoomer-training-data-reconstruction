#!/usr/bin/env python3
"""Generate image-level single-swap tests against the current best 18.

This is the phase after Open Images-only, ablation, and mixed-ratio scoring. It
keeps the current best hill-climbed anchor set fixed by default, ranks likely
weak anchor slots, ranks replacement candidates from Open Images, COCO, or
other near-domain pools, and writes candidate submissions of the form:

    anchor - anchor_candidate_04 + replacement_candidate_07

The goal is not source diversity. The goal is to test whether 1 exact image can
improve the current formal-workstation anchor.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from mix_coco_openimages_domain import (
    DEFAULT_PROMPTS,
    Candidate,
    collect_candidates,
    featurize_clip,
    l2_normalize,
    make_contact_sheet,
    normalized,
)


DEFAULT_SWAP_ANCHOR = Path(
    "analysis/anchor_ablation_tests/ablate_anchor17_neutral_filler_035"
)


def candidate_rank(candidate: Candidate) -> int:
    try:
        return int(candidate.path.stem.split("_")[-1])
    except ValueError:
        return 999


def write_swap(
    output_dir: Path,
    anchor: list[Candidate],
    replacement: Candidate,
    weak_index: int,
    anchor_score: np.ndarray,
    anchor_redundancy: np.ndarray,
    replacement_score: float,
    replacement_anchor_sim: float,
    replacement_theme_score: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    rows = []

    for rank, candidate in enumerate(anchor):
        source_role = "anchor_kept"
        selected = candidate
        if rank == weak_index:
            source_role = "replacement"
            selected = replacement

        destination = output_dir / f"candidate_{rank:02d}.ras"
        shutil.copy2(selected.path, destination)
        output_paths.append(destination)
        row = {
            "rank": str(rank),
            "filename": destination.name,
            "source_role": source_role,
            "source_dataset": selected.source,
            "source_variant": selected.source_variant,
            "source_file": str(selected.path),
            "replaced_anchor_rank": str(weak_index) if rank == weak_index else "",
            "replaced_anchor_file": str(anchor[weak_index].path) if rank == weak_index else "",
            "logit": f"{selected.logit:.9f}",
            "sha256": selected.digest,
        }
        if rank == weak_index:
            row["anchor_weak_score"] = f"{float(anchor_score[weak_index]):.9f}"
            row["anchor_redundancy"] = f"{float(anchor_redundancy[weak_index]):.9f}"
            row["replacement_score"] = f"{replacement_score:.9f}"
            row["replacement_anchor_similarity"] = f"{replacement_anchor_sim:.9f}"
            row["replacement_theme_score"] = f"{replacement_theme_score:.9f}"
        for key, value in selected.metadata.items():
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


def default_device() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-dir", type=Path, default=DEFAULT_SWAP_ANCHOR)
    parser.add_argument(
        "--replacement-roots",
        nargs="+",
        type=Path,
        default=[Path("analysis/openimages_domain_candidate_subsets/analysis/openimages_domain_retrieval")],
    )
    parser.add_argument("--output-root", type=Path, default=Path("analysis/single_swap_tests"))
    parser.add_argument(
        "--replacement-include",
        nargs="*",
        default=["person_tie", "person_laptop", "person_desk", "person_book", "person_mobile"],
    )
    parser.add_argument("--weak-count", type=int, default=4)
    parser.add_argument("--replacement-count", type=int, default=12)
    parser.add_argument("--max-tests", type=int, default=48)
    parser.add_argument("--limit-per-replacement-dir", type=int, default=18)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--logit-weight", type=float, default=0.05)
    parser.add_argument("--theme-weight", type=float, default=0.25)
    parser.add_argument(
        "--redundancy-weight",
        type=float,
        default=0.10,
        help="Higher values mark visually redundant anchor images as weaker.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anchor = collect_candidates(args.anchor_dir, "coco_anchor", include_substrings=[], limit_per_dir=18)
    if len(anchor) != 18:
        raise SystemExit(f"anchor must contain exactly 18 candidates, got {len(anchor)} from {args.anchor_dir}")
    anchor = sorted(anchor, key=candidate_rank)

    replacements = []
    seen = {candidate.digest for candidate in anchor}
    for root in args.replacement_roots:
        for candidate in collect_candidates(
            root,
            root.name,
            include_substrings=args.replacement_include,
            limit_per_dir=args.limit_per_replacement_dir,
        ):
            if candidate.digest in seen:
                continue
            seen.add(candidate.digest)
            replacements.append(candidate)
    if not replacements:
        raise SystemExit(f"no replacement candidates found under: {args.replacement_roots}")

    candidates = anchor + replacements
    features, text_features = featurize_clip(
        [candidate.path for candidate in candidates],
        DEFAULT_PROMPTS,
        args.model,
        args.device,
        args.batch_size,
    )
    features = l2_normalize(features)
    anchor_features = features[: len(anchor)]
    replacement_features = features[len(anchor) :]

    anchor_center = anchor_features.mean(axis=0)
    anchor_center = anchor_center / max(float(np.linalg.norm(anchor_center)), 1e-12)
    anchor_sim = features @ anchor_center
    theme_score = (features @ text_features.T).max(axis=1)
    logits = np.asarray([candidate.logit for candidate in candidates], dtype=np.float32)
    score = anchor_sim + args.theme_weight * theme_score + args.logit_weight * normalized(logits)

    anchor_similarity = anchor_features @ anchor_features.T
    np.fill_diagonal(anchor_similarity, -1.0)
    anchor_redundancy = anchor_similarity.max(axis=1)
    anchor_weak_score = score[: len(anchor)] - args.redundancy_weight * anchor_redundancy
    weak_indices = np.argsort(anchor_weak_score)[: args.weak_count].astype(int).tolist()

    replacement_offset = len(anchor)
    replacement_indices = (
        np.argsort(-score[replacement_offset:])[: args.replacement_count].astype(int) + replacement_offset
    ).tolist()

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    tests_written = 0
    for weak_index in weak_indices:
        for replacement_index in replacement_indices:
            replacement = candidates[replacement_index]
            replacement_local_index = replacement_index - replacement_offset
            name = (
                f"swap_anchor{weak_index:02d}_with_"
                f"{replacement.source}_{replacement_local_index:03d}"
            )
            output_dir = args.output_root / name
            write_swap(
                output_dir,
                anchor,
                replacement,
                weak_index,
                anchor_weak_score,
                anchor_redundancy,
                float(score[replacement_index]),
                float(anchor_sim[replacement_index]),
                float(theme_score[replacement_index]),
            )
            summary_rows.append(
                {
                    "variant": name,
                    "output_dir": str(output_dir),
                    "replaced_anchor_rank": weak_index,
                    "replaced_anchor_file": str(anchor[weak_index].path),
                    "replacement_rank": replacement_local_index,
                    "replacement_file": str(replacement.path),
                    "replacement_source": replacement.source,
                    "replacement_variant": replacement.source_variant,
                    "anchor_weak_score": f"{float(anchor_weak_score[weak_index]):.9f}",
                    "anchor_redundancy": f"{float(anchor_redundancy[weak_index]):.9f}",
                    "replacement_score": f"{float(score[replacement_index]):.9f}",
                    "replacement_anchor_similarity": f"{float(anchor_sim[replacement_index]):.9f}",
                    "replacement_theme_score": f"{float(theme_score[replacement_index]):.9f}",
                    "replacement_logit": f"{float(logits[replacement_index]):.9f}",
                }
            )
            tests_written += 1
            if tests_written >= args.max_tests:
                break
        if tests_written >= args.max_tests:
            break

    summary = args.output_root / "single_swap_summary.csv"
    with summary.open("w", newline="") as file:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"anchor candidates: {len(anchor)}")
    print(f"replacement candidates: {len(replacements)}")
    print(f"weak anchor ranks: {weak_indices}")
    print(f"top replacement count: {len(replacement_indices)}")
    print(f"single-swap tests written: {tests_written}")
    print(f"wrote summary: {summary}")


if __name__ == "__main__":
    main()
