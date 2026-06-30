"""Shared orchestration helpers for multi-split experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .performance import MultiSolverDataset

SEED_DIR_PATTERN = re.compile(r"^seed(\d+)$")
GAP_METRICS = ("gap_cls_solved", "gap_cls_par2")


def discover_seed_dirs(splits_dir: Path) -> list[tuple[int, Path]]:
    """Return valid ``seedN`` split directories sorted by numeric seed."""
    entries: list[tuple[int, Path]] = []
    for subdir in Path(splits_dir).iterdir():
        match = SEED_DIR_PATTERN.fullmatch(subdir.name)
        if (
            subdir.is_dir()
            and match
            and (subdir / "train.json").is_file()
            and (subdir / "test.json").is_file()
        ):
            entries.append((int(match.group(1)), subdir))
    return sorted(entries)


def select_seed_dirs(
    entries: Iterable[tuple[int, Path]],
    requested_seeds: Iterable[int] | None,
) -> list[tuple[int, Path]]:
    """Filter seed entries while rejecting requested seeds that do not exist."""
    entries = list(entries)
    if requested_seeds is None:
        return entries
    requested = set(requested_seeds)
    selected = [(seed, path) for seed, path in entries if seed in requested]
    missing = requested - {seed for seed, _ in selected}
    if missing:
        raise ValueError(f"Requested seeds not found: {sorted(missing)}")
    return selected


def rebase_performance_data(
    dataset: MultiSolverDataset,
    benchmark_root: Path,
) -> MultiSolverDataset:
    """Resolve relative instance paths beneath a benchmark root."""
    rebased = {str(benchmark_root / path): dataset[path] for path in dataset.keys()}
    return MultiSolverDataset(
        rebased,
        dataset.get_solver_id_dict(),
        dataset.get_timeout(),
    )


def aggregate_gap_metrics(
    seed_results: list[dict[str, Any]],
    splits: Iterable[str],
) -> dict[str, dict[str, float]]:
    """Compute population mean/std for common gap-closed metrics."""
    aggregated: dict[str, dict[str, float]] = {}
    for split in splits:
        metrics = [result[f"{split}_metrics"] for result in seed_results]
        aggregated[split] = {}
        for metric in GAP_METRICS:
            values = [entry[metric] for entry in metrics]
            aggregated[split][f"{metric}_mean"] = float(np.mean(values))
            aggregated[split][f"{metric}_std"] = float(np.std(values))
    return aggregated


def to_json_value(value: Any) -> Any:
    """Recursively convert NumPy values to JSON-native values."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    return value


def write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write an experiment summary using stable JSON formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_json_value(results), handle, indent=2)
