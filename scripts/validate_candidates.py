#!/usr/bin/env python3
"""Validate generated candidates with the original classifier executable.

This is for local use in the challenge repo, not Colab. It answers whether
saved .ras files still produce the target binary output after all file-format
and preprocessing effects.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


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
        required=True,
        help="Directory containing generated .ras files.",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Glob pattern. Defaults to candidate_*.ras, then final_*.ras if no candidates exist.",
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        default=Path("executables/linux-x86_64/classifier"),
    )
    parser.add_argument("--target-output", default="0")
    parser.add_argument("--expect-count", type=int, default=18)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_candidate_paths(args.candidate_dir, args.pattern)
    if not paths:
        pattern = args.pattern or "candidate_*.ras or final_*.ras"
        raise SystemExit(f"no files matched {args.candidate_dir} / {pattern}")
    if args.expect_count is not None and len(paths) != args.expect_count:
        raise SystemExit(f"expected {args.expect_count} files, matched {len(paths)}")

    rows = []
    accepted = 0
    for path in paths:
        result = subprocess.run(
            [str(args.classifier), str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout.strip()
        ok = result.returncode == 0 and output == args.target_output
        accepted += int(ok)
        rows.append((path.name, result.returncode, output, ok))
        print(f"{path.name}: output={output!r} exit={result.returncode} target={ok}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "exit_code", "stdout", "target_match"])
            writer.writerows(rows)
        print(f"wrote: {args.csv}")

    print(f"target matches: {accepted}/{len(paths)}")
    if accepted != len(paths):
        raise SystemExit(f"binary target validation failed: {accepted}/{len(paths)} matched")


if __name__ == "__main__":
    main()
