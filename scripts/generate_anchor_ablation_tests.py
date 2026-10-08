#!/usr/bin/env python3
"""Generate leave-one-out style ablation tests with controlled fillers.

Every challenge submission needs 18 images, so "remove candidate_i" must be
implemented as:

    anchor - candidate_i + filler

This script uses the current hill-climbed ablation winner as the default anchor,
ranks likely weak anchor slots by CLIP/theme/logit/redundancy proxies, then
writes ablation submissions with two filler types:

    strong filler: high-ranked same-domain replacement candidate
    neutral filler: middle-ranked same-domain replacement candidate

If replacing the same anchor slot with both filler types improves the hidden
score, that anchor image is probably harmful or weak. If both drop, it is
probably a core contributor.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from generate_single_swap_tests import DEFAULT_SWAP_ANCHOR, candidate_rank
from mix_coco_openimages_domain import (
    DEFAULT_PROMPTS,
    Candidate,
    collect_candidates,
    featurize_clip,
    l2_normalize,
    make_contact_sheet,
    normalized,
)


def default_device() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def write_ablation(
    output_dir: Path,
    anchor: list[Candidate],
    weak_index: int,
    filler: Candidate,
    filler_kind: str,
    anchor_weak_score: np.ndarray,
    anchor_redundancy: np.ndarray,
    score: np.ndarray,
    anchor_sim: np.ndarray,
    theme_score: np.ndarray,
    filler_global_index: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    rows = []
    for rank, candidate in enumerate(anchor):
        selected = filler if rank == weak_index else candidate
        role = filler_kind if rank == weak_index else "anchor_kept"
        destination = output_dir / f"candidate_{rank:02d}.ras"
        shutil.copy2(selected.path, destination)
        output_paths.append(destination)

        row = {
            "rank": str(rank),
            "filename": destination.name,
            "source_role": role,
            "source_dataset": selected.source,
            "source_variant": selected.source_variant,
            "source_file": str(selected.path),
            "ablation_anchor_rank": str(weak_index) if rank == weak_index else "",
            "ablation_anchor_file": str(anchor[weak_index].path) if rank == weak_index else "",
            "filler_kind": filler_kind if rank == weak_index else "",
            "logit": f"{selected.logit:.9f}",
            "sha256": selected.digest,
        }
        if rank == weak_index:
            row["anchor_weak_score"] = f"{float(anchor_weak_score[weak_index]):.9f}"
            row["anchor_redundancy"] = f"{float(anchor_redundancy[weak_index]):.9f}"
            row["filler_score"] = f"{float(score[filler_global_index]):.9f}"
            row["filler_anchor_similarity"] = f"{float(anchor_sim[filler_global_index]):.9f}"
            row["filler_theme_score"] = f"{float(theme_score[filler_global_index]):.9f}"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-dir", type=Path, default=DEFAULT_SWAP_ANCHOR)
    parser.add_argument(
        "--replacement-roots",
        nargs="+",
        type=Path,
        default=[
            Path("analysis/openimages_domain_candidate_subsets/analysis/openimages_domain_retrieval"),
            Path(
                "analysis/person_tie_office_refinement_candidate_subsets/analysis/"
                "person_tie_office_refinement/semantic_clip/person_tie_laptop_chair"
            ),
        ],
    )
    parser.add_argument("--output-root", type=Path, default=Path("analysis/anchor_ablation_tests"))
    parser.add_argument(
        "--replacement-include",
        nargs="*",
        default=["openimages", "person_tie", "clip"],
    )
    parser.add_argument("--weak-count", type=int, default=4)
    parser.add_argument("--strong-fillers", type=int, default=2)
    parser.add_argument("--neutral-fillers", type=int, default=2)
    parser.add_argument("--max-tests", type=int, default=24)
    parser.add_argument("--limit-per-replacement-dir", type=int, default=18)
    parser.add_argument(
        "--max-replacements-per-root",
        type=int,
        default=0,
        help="Optional cap after collecting each replacement root. 0 keeps all candidates.",
    )
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--logit-weight", type=float, default=0.05)
    parser.add_argument("--theme-weight", type=float, default=0.25)
    parser.add_argument("--redundancy-weight", type=float, default=0.10)
    parser.add_argument(
        "--priority-include",
        nargs="*",
        default=[],
        help="Optional source path/name substrings that get a filler ranking boost.",
    )
    parser.add_argument(
        "--priority-weight",
        type=float,
        default=0.0,
        help="Score boost for replacement candidates whose source path/name matches --priority-include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"loading anchor: {args.anchor_dir}", flush=True)
    anchor = collect_candidates(args.anchor_dir, "openimages_anchor", include_substrings=[], limit_per_dir=18)
    if len(anchor) != 18:
        raise SystemExit(f"anchor must contain exactly 18 candidates, got {len(anchor)} from {args.anchor_dir}")
    anchor = sorted(anchor, key=candidate_rank)
    print(f"anchor candidates: {len(anchor)}", flush=True)

    replacements = []
    seen = {candidate.digest for candidate in anchor}
    for root in args.replacement_roots:
        print(f"collecting replacements from: {root}", flush=True)
        root_candidates = collect_candidates(
            root,
            root.name,
            include_substrings=args.replacement_include,
            limit_per_dir=args.limit_per_replacement_dir,
        )
        if args.max_replacements_per_root > 0:
            root_candidates = root_candidates[: args.max_replacements_per_root]
        print(f"  collected before de-dupe/cap: {len(root_candidates)}", flush=True)
        root_added = 0
        for candidate in root_candidates:
            if candidate.digest in seen:
                continue
            seen.add(candidate.digest)
            replacements.append(candidate)
            root_added += 1
        print(f"  added after de-dupe: {root_added}; total replacements: {len(replacements)}", flush=True)
    if not replacements:
        raise SystemExit(f"no replacement candidates found under: {args.replacement_roots}")

    candidates = anchor + replacements
    print(
        f"featurizing {len(candidates)} images with {args.model} on {args.device}; "
        f"batch_size={args.batch_size}",
        flush=True,
    )
    features, text_features = featurize_clip(
        [candidate.path for candidate in candidates],
        DEFAULT_PROMPTS,
        args.model,
        args.device,
        args.batch_size,
    )
    print("finished CLIP featurization", flush=True)
    features = l2_normalize(features)
    anchor_features = features[: len(anchor)]
    anchor_center = anchor_features.mean(axis=0)
    anchor_center = anchor_center / max(float(np.linalg.norm(anchor_center)), 1e-12)

    anchor_sim = features @ anchor_center
    theme_score = (features @ text_features.T).max(axis=1)
    logits = np.asarray([candidate.logit for candidate in candidates], dtype=np.float32)
    priority_score = np.zeros(len(candidates), dtype=np.float32)
    if args.priority_include and args.priority_weight != 0:
        needles = [value.lower() for value in args.priority_include]
        for index, candidate in enumerate(candidates):
            if index < len(anchor):
                continue
            haystack = " ".join(
                [
                    str(candidate.path),
                    candidate.source,
                    candidate.source_variant,
                    " ".join(f"{key}={value}" for key, value in candidate.metadata.items()),
                ]
            ).lower()
            if any(needle in haystack for needle in needles):
                priority_score[index] = 1.0
    score = (
        anchor_sim
        + args.theme_weight * theme_score
        + args.logit_weight * normalized(logits)
        + args.priority_weight * priority_score
    )

    anchor_similarity = anchor_features @ anchor_features.T
    np.fill_diagonal(anchor_similarity, -1.0)
    anchor_redundancy = anchor_similarity.max(axis=1)
    anchor_weak_score = score[: len(anchor)] - args.redundancy_weight * anchor_redundancy
    weak_indices = np.argsort(anchor_weak_score)[: args.weak_count].astype(int).tolist()

    replacement_offset = len(anchor)
    ranked_replacements = (
        np.argsort(-score[replacement_offset:]).astype(int) + replacement_offset
    ).tolist()
    strong_indices = ranked_replacements[: args.strong_fillers]

    neutral_indices = []
    if args.neutral_fillers > 0:
        middle = len(ranked_replacements) // 2
        half = args.neutral_fillers // 2
        start = max(0, middle - half)
        neutral_indices = ranked_replacements[start : start + args.neutral_fillers]

    filler_jobs = [("strong_filler", index) for index in strong_indices]
    filler_jobs += [("neutral_filler", index) for index in neutral_indices]
    if not filler_jobs:
        raise SystemExit("no fillers selected")
    print(f"weak anchor ranks: {weak_indices}", flush=True)
    print(f"strong fillers: {[idx - replacement_offset for idx in strong_indices]}", flush=True)
    print(f"neutral fillers: {[idx - replacement_offset for idx in neutral_indices]}", flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    tests_written = 0
    for weak_index in weak_indices:
        for filler_kind, filler_index in filler_jobs:
            filler = candidates[filler_index]
            filler_local_index = filler_index - replacement_offset
            name = f"ablate_anchor{weak_index:02d}_{filler_kind}_{filler_local_index:03d}"
            output_dir = args.output_root / name
            print(f"writing {tests_written + 1}/{args.max_tests}: {name}", flush=True)
            write_ablation(
                output_dir,
                anchor,
                weak_index,
                filler,
                filler_kind,
                anchor_weak_score,
                anchor_redundancy,
                score,
                anchor_sim,
                theme_score,
                filler_index,
            )
            summary_rows.append(
                {
                    "variant": name,
                    "output_dir": str(output_dir),
                    "ablation_anchor_rank": weak_index,
                    "ablation_anchor_file": str(anchor[weak_index].path),
                    "filler_kind": filler_kind,
                    "filler_rank": filler_local_index,
                    "filler_file": str(filler.path),
                    "filler_source": filler.source,
                    "filler_variant": filler.source_variant,
                    "anchor_weak_score": f"{float(anchor_weak_score[weak_index]):.9f}",
                    "anchor_redundancy": f"{float(anchor_redundancy[weak_index]):.9f}",
                    "filler_score": f"{float(score[filler_index]):.9f}",
                    "filler_anchor_similarity": f"{float(anchor_sim[filler_index]):.9f}",
                    "filler_theme_score": f"{float(theme_score[filler_index]):.9f}",
                    "filler_priority_score": f"{float(priority_score[filler_index]):.9f}",
                    "filler_logit": f"{float(logits[filler_index]):.9f}",
                }
            )
            tests_written += 1
            if tests_written >= args.max_tests:
                break
        if tests_written >= args.max_tests:
            break

    summary = args.output_root / "anchor_ablation_summary.csv"
    with summary.open("w", newline="") as file:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"anchor candidates: {len(anchor)}", flush=True)
    print(f"replacement candidates: {len(replacements)}", flush=True)
    print(f"ablation tests written: {tests_written}", flush=True)
    print(f"wrote summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
