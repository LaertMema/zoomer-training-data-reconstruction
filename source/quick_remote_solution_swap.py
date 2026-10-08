#!/usr/bin/env python3
"""Quickly swap a Zoomer candidate set into the remote solution folder.

This is optimized for leaderboard iteration:

1. Copy one local candidate folder to a remote staging directory.
2. Replace /home/user/solution/*.ras with those 18 files.
3. Record which label is currently staged, outside solution/.
4. Run task check on the remote and save the output locally.

It does not write marker files into solution/ because the challenge expects the
solution directory to contain the submitted images.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


LOCAL_RESULTS = Path("analysis/remote_submission_results.csv")
LOCAL_LOG_DIR = Path("analysis/remote_submission_logs")


def quote(value: str) -> str:
    return shlex.quote(value)


def run(command: list[str], dry_run: bool, capture: bool = False) -> subprocess.CompletedProcess[str] | None:
    print("$ " + " ".join(quote(part) for part in command))
    if dry_run:
        return None
    return subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=capture,
    )


def remote_join(*parts: str) -> str:
    return "/" + "/".join(part.strip("/") for part in parts if part.strip("/") != "")


def sanitize_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)


def append_result(
    timestamp: str,
    label: str,
    candidate_dir: Path,
    remote_stage: str,
    returncode: int | str,
    log_path: Path,
) -> None:
    LOCAL_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    exists = LOCAL_RESULTS.exists()
    with LOCAL_RESULTS.open("a", newline="") as file:
        writer = csv.writer(file)
        if not exists:
            writer.writerow(
                [
                    "timestamp",
                    "label",
                    "candidate_dir",
                    "remote_stage",
                    "returncode",
                    "log_path",
                    "score_or_metric_notes",
                ]
            )
        writer.writerow([timestamp, label, str(candidate_dir), remote_stage, returncode, str(log_path), ""])


def status(args: argparse.Namespace) -> int:
    remote_root = args.remote_root.rstrip("/")
    state_file = remote_join(remote_root, ".zoomer_current_solution")
    solution_dir = remote_join(remote_root, "solution")
    command = (
        "set -e; "
        f"echo 'remote root: {quote(remote_root)}'; "
        f"echo 'current state:'; "
        f"cat {quote(state_file)} 2>/dev/null || echo 'unknown'; "
        f"echo 'solution ras count:'; "
        f"find {quote(solution_dir)} -maxdepth 1 -type f -name '*.ras' | wc -l"
    )
    result = run(["ssh", args.remote, "bash", "-lc", command], args.dry_run, capture=False)
    return 0 if result is None else result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast remote /home/user/solution swapper with label tracking."
    )
    parser.add_argument(
        "candidate_dir",
        nargs="?",
        type=Path,
        help="Local directory containing exactly 18 candidate_*.ras files.",
    )
    parser.add_argument("--remote", default="zoomer@pine")
    parser.add_argument("--remote-root", default="/home/user")
    parser.add_argument("--label", default=None)
    parser.add_argument("--pattern", default="candidate_*.ras")
    parser.add_argument("--check-cmd", default="task check")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--status", action="store_true", help="Only show the currently staged remote label.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.status:
        return status(args)

    if args.candidate_dir is None:
        raise SystemExit("candidate_dir is required unless --status is used")

    candidate_dir = args.candidate_dir
    paths = sorted(candidate_dir.glob(args.pattern))
    if not candidate_dir.is_dir():
        raise SystemExit(f"candidate directory does not exist: {candidate_dir}")
    if len(paths) != 18:
        raise SystemExit(f"expected 18 files matching {args.pattern}, found {len(paths)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = sanitize_label(args.label or candidate_dir.name)
    remote_root = args.remote_root.rstrip("/")
    remote_stage = remote_join(remote_root, ".zoomer_submissions", f"{timestamp}_{label}")
    remote_solution = remote_join(remote_root, "solution")
    state_file = remote_join(remote_root, ".zoomer_current_solution")
    local_log = LOCAL_LOG_DIR / f"{timestamp}_{label}.txt"

    print(f"label:      {label}")
    print(f"candidate:  {candidate_dir}")
    print(f"remote:     {args.remote}")
    print(f"solution:   {remote_solution}")
    print(f"state file: {state_file}")
    print()

    mkdir_result = run(["ssh", args.remote, "mkdir", "-p", remote_stage], args.dry_run)
    if mkdir_result is not None and mkdir_result.returncode != 0:
        return mkdir_result.returncode

    rsync_result = run(
        [
            "rsync",
            "-az",
            "--delete",
            "--include",
            args.pattern,
            "--exclude",
            "*",
            str(candidate_dir) + "/",
            f"{args.remote}:{remote_stage}/",
        ],
        args.dry_run,
    )
    if rsync_result is not None and rsync_result.returncode != 0:
        return rsync_result.returncode

    remote_commands = [
        "set -e",
        f"cd {quote(remote_root)}",
        f"test $(find {quote(remote_stage)} -maxdepth 1 -type f -name '*.ras' | wc -l) -eq 18",
        f"mkdir -p {quote(remote_solution)}",
        f"find {quote(remote_solution)} -maxdepth 1 -type f -name '*.ras' -delete",
        f"cp {quote(remote_stage)}/*.ras {quote(remote_solution)}/",
        f"test $(find {quote(remote_solution)} -maxdepth 1 -type f -name '*.ras' | wc -l) -eq 18",
        (
            "printf '%s\\n' "
            f"{quote('label=' + label)} "
            f"{quote('timestamp=' + timestamp)} "
            f"{quote('stage=' + remote_stage)} "
            f"{quote('solution=' + remote_solution)} "
            f"> {quote(state_file)}"
        ),
        f"echo 'CURRENT_ZOOMER_LABEL={quote(label)}'",
    ]
    if not args.no_check:
        remote_commands.append(args.check_cmd)

    result = run(
        ["ssh", args.remote, "bash", "-lc", "; ".join(remote_commands)],
        args.dry_run,
        capture=True,
    )

    if args.dry_run:
        append_result(timestamp, label, candidate_dir, remote_stage, "dry-run", local_log)
        return 0

    assert result is not None
    output = (result.stdout or "") + (result.stderr or "")
    LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    local_log.write_text(output)

    print(output, end="" if output.endswith("\n") else "\n")
    append_result(timestamp, label, candidate_dir, remote_stage, result.returncode, local_log)

    print()
    print(f"wrote local log: {local_log}")
    print(f"updated results: {LOCAL_RESULTS}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
