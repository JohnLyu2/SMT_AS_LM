"""CSV input/output shared by evaluation and fallback handling."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from smt_select.utils import normalize_path

CSV_HEADER = [
    "benchmark",
    "selected",
    "solved",
    "runtime",
    "solver_runtime",
    "overhead",
    "feature_fail",
]


def load_extraction_times_csv(path: Path) -> dict[str, float]:
    times: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                times[normalize_path(row["path"])] = float(row["time_sec"])
            except (KeyError, ValueError):
                continue
    return times


def load_failed_paths_from_extraction_times_csv(path: Path) -> list[str]:
    failed: list[str] = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and "failed" not in reader.fieldnames:
            return []
        for row in reader:
            if (row.get("failed") or "").strip() == "1":
                normalized = normalize_path((row.get("path") or "").strip())
                if normalized:
                    failed.append(normalized)
    return failed


def csv_benchmark(path: str, root: Path | None) -> str:
    if root is None:
        return path
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return path


def load_eval_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_eval_csv(
    path: Path,
    rows: Iterable[dict],
    header: list[str] | None = None,
) -> None:
    columns = header or CSV_HEADER
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row.get(column, "") for column in columns])
