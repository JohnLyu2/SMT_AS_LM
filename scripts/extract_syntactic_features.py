#!/usr/bin/env python3
"""Reproduce per-logic syntactic feature tables with Klammerhammer."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.extract_syntactic_csv import FIXED_COLUMNS, SYMBOLS
except ModuleNotFoundError:
    # Support direct execution as ``python scripts/extract_syntactic_features.py``.
    from extract_syntactic_csv import FIXED_COLUMNS, SYMBOLS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KLHM_TO_COLUMN = {
    "asserts_count": "assertsCount",
    "declare_fun_count": "declareFunCount",
    "declare_const_count": "declareConstCount",
    "declare_sort_count": "declareSortCount",
    "define_fun_count": "defineFunCount",
    "define_fun_rec_count": "defineFunRecCount",
    "constant_fun_count": "constantFunCount",
    "define_sort_count": "defineSortCount",
    "declare_datatype_count": "declareDatatypeCount",
    "max_term_depth": "maxTermDepth",
}


@dataclass(frozen=True)
class Extraction:
    relative_path: str
    elapsed_seconds: float
    failed: bool
    feature_values: list[int]
    error: str | None = None


def parse_klhm_output(stdout: str) -> list[int]:
    """Convert Klammerhammer's JSON output to the checked-in CSV column order."""
    payload = json.loads(stdout)
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError("expected Klammerhammer to return a two-element JSON array")

    syntactic, metadata = payload[0], payload[1]
    if not isinstance(syntactic, dict) or not isinstance(metadata, dict):
        raise ValueError("invalid Klammerhammer JSON objects")

    frequencies = syntactic.get("symbolFrequency")
    if not isinstance(frequencies, list) or len(frequencies) != len(SYMBOLS):
        raise ValueError(
            f"expected {len(SYMBOLS)} symbol frequencies, got "
            f"{len(frequencies) if isinstance(frequencies, list) else 'invalid data'}"
        )

    fixed = {
        "size": metadata.get("size", syntactic.get("normalizedSize", 0)),
        **{
            column: syntactic.get(klhm_key, 0)
            for column, klhm_key in KLHM_TO_COLUMN.items()
        },
    }
    return [int(fixed[column] or 0) for column in FIXED_COLUMNS] + [
        int(value or 0) for value in frequencies
    ]


def extract_one(
    relative_path: str,
    benchmark_root: Path,
    klhm: Path,
    timeout: float,
) -> Extraction:
    benchmark = benchmark_root / relative_path
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [str(klhm), str(benchmark)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        values = parse_klhm_output(result.stdout)
        return Extraction(
            relative_path,
            time.perf_counter() - started,
            False,
            values,
        )
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
        return Extraction(
            relative_path,
            time.perf_counter() - started,
            True,
            [0] * (len(FIXED_COLUMNS) + len(SYMBOLS)),
            str(exc),
        )


def paths_from_meta(meta_json: Path) -> list[str]:
    with meta_json.open(encoding="utf-8") as handle:
        entries = json.load(handle)
    paths = [entry["smtlib_path"] for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError(f"duplicate smtlib_path entries in {meta_json}")
    return paths


def write_logic_tables(output_dir: Path, rows: list[Extraction]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "features.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", *FIXED_COLUMNS, *SYMBOLS])
        for row in rows:
            writer.writerow([row.relative_path, *row.feature_values])

    with (output_dir / "extraction_times.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "time_sec", "failed"])
        for row in rows:
            writer.writerow(
                [row.relative_path, row.elapsed_seconds, int(row.failed)]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Klammerhammer features for one or more SMT logics.",
    )
    parser.add_argument(
        "--logic",
        nargs="+",
        help="Logic names to process; omit to process every JSON in --meta-dir.",
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw_data" / "meta_info",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=PROJECT_ROOT / "smtlib" / "non-incremental",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "syntactic",
    )
    parser.add_argument(
        "--klhm",
        type=Path,
        default=PROJECT_ROOT / "bin" / "klhm",
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not args.meta_dir.is_dir():
        parser.error(f"metadata directory not found: {args.meta_dir}")
    if not args.benchmark_root.is_dir():
        parser.error(f"benchmark root not found: {args.benchmark_root}")
    if not args.klhm.is_file():
        parser.error(f"Klammerhammer executable not found: {args.klhm}")

    logics = args.logic or sorted(path.stem for path in args.meta_dir.glob("*.json"))
    for logic in logics:
        meta_json = args.meta_dir / f"{logic}.json"
        if not meta_json.is_file():
            parser.error(f"metadata file not found: {meta_json}")
        paths = paths_from_meta(meta_json)

        def run(path: str) -> Extraction:
            return extract_one(path, args.benchmark_root, args.klhm, args.timeout)

        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            rows = list(executor.map(run, paths))

        write_logic_tables(args.output_dir / logic, rows)
        failures = [row for row in rows if row.failed]
        print(
            f"{logic}: wrote {len(rows)} rows to {args.output_dir / logic} "
            f"({len(failures)} failed)"
        )
        for row in failures[:10]:
            print(f"  failed: {row.relative_path}: {row.error}")


if __name__ == "__main__":
    main()
