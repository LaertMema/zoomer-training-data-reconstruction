#!/usr/bin/env python3
"""Rank and select candidates using classifier score plus feature diversity.

Use this when we have many generated .ras files. It greedily selects candidates
that are both high-logit and far apart in the reconstructed model's avgpool
feature space, which is a practical way to reduce near-duplicates.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from preprocess_compare import preprocess
from reconstruct_model import build_resnet18, load_flat_weights, reconstruct_state_dict


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


def featurize(model: torch.nn.Module, paths: list[Path], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    hook = FeatureHook(model.avgpool)
    logits = []
    features = []
    try:
        for path in paths:
            tensor = preprocess(path)
            x = torch.from_numpy(tensor.copy()).unsqueeze(0).to(device)
            with torch.inference_mode():
                logit = model(x).reshape(()).item()
                if hook.value is None:
                    raise RuntimeError("feature hook did not capture avgpool output")
                feature = F.normalize(hook.value, dim=1).squeeze(0).cpu().numpy()
            logits.append(logit)
            features.append(feature)
    finally:
        hook.close()
    return np.asarray(logits, dtype=np.float32), np.vstack(features).astype(np.float32)


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
        scores = []
        for idx in remaining:
            max_similarity = float((features[idx] @ selected_features.T).max())
            score_bonus = 0.05 * float(logits[idx])
            scores.append((max_similarity - score_bonus, idx))
        selected.append(min(scores)[1])
    return selected


def resolve_candidate_paths(candidate_dir: Path, pattern: str | None) -> list[Path]:
    if pattern is not None:
        return sorted(candidate_dir.glob(pattern))
    paths = sorted(candidate_dir.glob("candidate_*.ras"))
    if paths:
        return paths
    return sorted(candidate_dir.glob("final_*.ras"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument(
        "--pattern",
        default=None,
        help="Glob pattern. Defaults to candidate_*.ras, then final_*.ras if no candidates exist.",
    )
    parser.add_argument("--count", type=int, default=18)
    parser.add_argument("--expect-count", type=int, default=None)
    parser.add_argument("--min-logit", type=float, default=1.0)
    parser.add_argument("--weights", type=Path, default=Path("analysis/resnet18_weights.bin"))
    parser.add_argument("--checkpoint", type=Path, default=Path("analysis/reconstructed_resnet18.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/selected_candidates"))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_candidate_paths(args.candidate_dir, args.pattern)
    if not paths:
        pattern = args.pattern or "candidate_*.ras or final_*.ras"
        raise SystemExit(f"no files matched {args.candidate_dir} / {pattern}")
    if args.expect_count is not None and len(paths) != args.expect_count:
        raise SystemExit(f"expected {args.expect_count} input files, matched {len(paths)}")

    device = torch.device(args.device)
    model = load_model(args, device)
    logits, features = featurize(model, paths, device)
    selected = greedy_select(logits, features, args.count, args.min_logit)
    if len(selected) != args.count:
        raise SystemExit(f"expected to select {args.count} files, selected {len(selected)}")

    similarity = features @ features.T
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rank, idx in enumerate(selected):
        destination = args.output_dir / f"candidate_{rank:02d}.ras"
        shutil.copy2(paths[idx], destination)
        selected_others = [j for j in selected if j != idx]
        max_selected_similarity = (
            float(similarity[idx, selected_others].max()) if selected_others else 0.0
        )
        rows.append((rank, paths[idx].name, float(logits[idx]), max_selected_similarity))
        print(
            f"{rank:02d} {paths[idx].name} "
            f"logit={logits[idx]:.6f} "
            f"max_selected_cos={max_selected_similarity:.6f}"
        )

    csv_path = args.csv or (args.output_dir / "selection.csv")
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["rank", "filename", "logit", "max_selected_cosine"])
        writer.writerows(rows)
    print(f"wrote selected files to: {args.output_dir}")
    print(f"wrote: {csv_path}")


if __name__ == "__main__":
    main()
