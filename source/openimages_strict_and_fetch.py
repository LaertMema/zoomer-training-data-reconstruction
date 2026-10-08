#!/usr/bin/env python3
"""Fetch Open Images samples that contain all requested labels.

FiftyOne's class filter is useful, but for multi-label target combinations we
want stricter behavior: inspect Open Images annotation metadata first, intersect
image IDs that contain every required class, and only download those images.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


CLASS_DESCRIPTION_URLS = [
    "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv",
    "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
]

BBOX_URLS = {
    "validation": [
        "https://storage.googleapis.com/openimages/v6/oidv6-validation-annotations-bbox.csv",
        "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    ],
    "train": [
        "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
        "https://storage.googleapis.com/openimages/v5/train-annotations-bbox.csv",
    ],
    "test": [
        "https://storage.googleapis.com/openimages/v6/oidv6-test-annotations-bbox.csv",
        "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
    ],
}

IMAGE_URLS = {
    "validation": [
        "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    ],
    "train": [
        "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
    ],
    "test": [
        "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", required=True, help="Pipe-delimited display names, e.g. Person|Laptop|Tie")
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--metadata-cache", type=Path, default=Path("/content/openimages_metadata_cache"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument(
        "--min-label-matches",
        type=int,
        default=0,
        help="Minimum number of requested labels that must be present. 0 means require all labels.",
    )
    parser.add_argument(
        "--always-require",
        nargs="*",
        default=[],
        help="Requested display labels that must always be present, e.g. Laptop.",
    )
    return parser.parse_args()


def download_first(urls: list[str], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    last_error = None
    for url in urls:
        try:
            print("metadata download:", url, "->", destination, flush=True)
            urllib.request.urlretrieve(url, destination)
            if destination.exists() and destination.stat().st_size > 0:
                return destination
        except (urllib.error.URLError, OSError) as error:
            last_error = error
            if destination.exists():
                destination.unlink()
            print("metadata URL failed:", url, repr(error), flush=True)
    raise RuntimeError(f"all metadata URLs failed for {destination}: {last_error!r}")


def load_class_map(cache: Path) -> dict[str, str]:
    csv_path = download_first(CLASS_DESCRIPTION_URLS, cache / "class_descriptions.csv")
    mapping: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 2:
                continue
            label_name, display_name = row[0], row[1]
            mapping[display_name.lower()] = label_name
    return mapping


def resolve_labels(classes: list[str], class_map: dict[str, str]) -> dict[str, str]:
    resolved = {}
    for class_name in classes:
        key = class_name.lower()
        if key not in class_map:
            raise SystemExit(f"Open Images class not found: {class_name!r}")
        resolved[class_name] = class_map[key]
    return resolved


def matching_image_ids(
    annotation_csv: Path,
    label_codes: list[str],
    min_matches: int,
    always_required: list[str],
) -> list[str]:
    required = set(label_codes)
    bit_for_label = {label: 1 << index for index, label in enumerate(label_codes)}
    always_mask = 0
    for label in always_required:
        always_mask |= bit_for_label[label]
    masks: dict[str, int] = {}
    chunksize = 1_000_000
    usecols = ["ImageID", "LabelName"]
    for chunk_index, chunk in enumerate(pd.read_csv(annotation_csv, usecols=usecols, chunksize=chunksize)):
        chunk = chunk[chunk["LabelName"].isin(required)]
        if chunk.empty:
            continue
        for image_id, label_name in chunk.drop_duplicates().itertuples(index=False):
            masks[image_id] = masks.get(image_id, 0) | bit_for_label[label_name]
        if chunk_index % 10 == 0:
            matched = sum(
                1
                for mask in masks.values()
                if (mask & always_mask) == always_mask and mask.bit_count() >= min_matches
            )
            print(
                f"annotation chunks={chunk_index + 1} partial_images={len(masks)} matches={matched}",
                flush=True,
            )
    return [
        image_id
        for image_id, mask in masks.items()
        if (mask & always_mask) == always_mask and mask.bit_count() >= min_matches
    ]


def load_image_urls(image_csv: Path, image_ids: set[str]) -> list[dict[str, str]]:
    rows = []
    for chunk in pd.read_csv(image_csv, chunksize=500_000):
        chunk = chunk[chunk["ImageID"].isin(image_ids)]
        if chunk.empty:
            continue
        for row in chunk.to_dict("records"):
            rows.append(row)
    return rows


def download_image(row: dict[str, str], export_dir: Path, timeout: int) -> bool:
    image_id = row["ImageID"]
    destination = export_dir / f"{image_id}.jpg"
    urls = [
        str(row.get("OriginalURL", "") or ""),
        str(row.get("Thumbnail300KURL", "") or ""),
    ]
    urls = [url for url in urls if url.startswith("http")]
    for url in urls:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
            if data:
                destination.write_bytes(data)
                return True
        except Exception as error:  # noqa: BLE001 - keep batch download moving
            print("image download failed:", image_id, url, repr(error), flush=True)
            time.sleep(0.1)
    return False


def main() -> int:
    args = parse_args()
    classes = [value.strip() for value in args.classes.split("|") if value.strip()]
    if not classes:
        raise SystemExit("no classes provided")

    args.export_dir.mkdir(parents=True, exist_ok=True)
    for old_file in args.export_dir.iterdir():
        if old_file.is_file() and old_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            old_file.unlink()

    class_map = load_class_map(args.metadata_cache)
    resolved = resolve_labels(classes, class_map)
    print("target:", args.target_label, flush=True)
    print("required display classes:", classes, flush=True)
    print("required label codes:", resolved, flush=True)

    annotation_csv = download_first(BBOX_URLS[args.split], args.metadata_cache / f"{args.split}_bbox.csv")
    image_csv = download_first(IMAGE_URLS[args.split], args.metadata_cache / f"{args.split}_images.csv")

    label_codes = list(resolved.values())
    min_matches = args.min_label_matches or len(label_codes)
    min_matches = max(1, min(min_matches, len(label_codes)))
    always_required_codes = [resolved[name] for name in args.always_require if name in resolved]
    missing_always_required = [name for name in args.always_require if name not in resolved]
    if missing_always_required:
        print("always-required labels not in this target; ignoring:", missing_always_required, flush=True)
    print("minimum label matches:", min_matches, flush=True)
    print("always-required display labels:", [name for name in args.always_require if name in resolved], flush=True)

    matches = matching_image_ids(
        annotation_csv,
        label_codes,
        min_matches=min_matches,
        always_required=always_required_codes,
    )
    print("metadata-matching image IDs:", len(matches), flush=True)
    if not matches:
        raise SystemExit(3)

    rng = random.Random(args.seed)
    rng.shuffle(matches)
    if args.max_samples > 0:
        matches = matches[: args.max_samples]
    rows = load_image_urls(image_csv, set(matches))
    rng.shuffle(rows)
    print("image URL rows:", len(rows), flush=True)

    copied = 0
    for row in rows:
        if download_image(row, args.export_dir, args.timeout):
            copied += 1
            if copied % 100 == 0:
                print("downloaded strict AND images:", copied, flush=True)
        if args.max_samples > 0 and copied >= args.max_samples:
            break

    print("final copied strict AND", copied, "to", args.export_dir, flush=True)
    if copied == 0:
        raise SystemExit(3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
