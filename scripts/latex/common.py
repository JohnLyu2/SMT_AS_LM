"""Shared helpers for generating paper tables from experiment summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def latex_escape(value: str) -> str:
    """Escape plain text for use in a LaTeX table cell."""
    for old, new in (
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ):
        value = value.replace(old, new)
    return value


def load_summary(path: Path) -> dict[str, Any] | None:
    """Load a summary JSON file, returning ``None`` for absent/invalid files."""
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def test_metric_mean_std(
    summary_path: Path,
    metric: str,
) -> tuple[float | None, float | None]:
    """Read an aggregated test metric, falling back to per-seed values."""
    summary = load_summary(summary_path)
    if summary is None:
        return None, None
    aggregated = summary.get("aggregated", {}).get("test", {})
    mean_key, std_key = f"{metric}_mean", f"{metric}_std"
    if mean_key in aggregated and std_key in aggregated:
        return float(aggregated[mean_key]), float(aggregated[std_key])
    values = [
        float(seed["test_metrics"][metric])
        for seed in summary.get("seeds") or []
        if metric in seed.get("test_metrics", {})
    ]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    std = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
    return mean, std


def collect_summary_logics(root: Path) -> set[str]:
    """Return child directory names containing ``summary.json``."""
    if not root.is_dir():
        return set()
    return {
        child.name
        for child in root.iterdir()
        if child.is_dir() and (child / "summary.json").is_file()
    }
