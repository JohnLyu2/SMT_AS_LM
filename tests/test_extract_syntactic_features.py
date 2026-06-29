"""Tests for reproducible Klammerhammer feature extraction."""

import csv
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.extract_syntactic_csv import FIXED_COLUMNS, SYMBOLS
from scripts.extract_syntactic_features import (
    Extraction,
    extract_one,
    parse_klhm_output,
    paths_from_meta,
    write_logic_tables,
)


def klhm_payload() -> str:
    frequencies = [0] * len(SYMBOLS)
    frequencies[SYMBOLS.index("Int")] = 3
    return json.dumps(
        [
            {
                "normalizedSize": 98,
                "assertsCount": 2,
                "declareConstCount": 1,
                "maxTermDepth": 4,
                "symbolFrequency": frequencies,
            },
            {"logic": "QF_LIA", "size": 100},
        ]
    )


def test_parse_klhm_output_uses_expected_column_order():
    values = parse_klhm_output(klhm_payload())

    assert len(values) == len(FIXED_COLUMNS) + len(SYMBOLS)
    assert values[FIXED_COLUMNS.index("size")] == 100
    assert values[FIXED_COLUMNS.index("asserts_count")] == 2
    assert values[FIXED_COLUMNS.index("declare_const_count")] == 1
    assert values[len(FIXED_COLUMNS) + SYMBOLS.index("Int")] == 3


def test_parse_klhm_output_rejects_wrong_symbol_count():
    payload = json.loads(klhm_payload())
    payload[0]["symbolFrequency"].pop()

    with pytest.raises(ValueError, match="symbol frequencies"):
        parse_klhm_output(json.dumps(payload))


def test_extract_one_marks_klhm_failure():
    error = subprocess.CalledProcessError(1, ["klhm", "example.smt2"])
    with patch(
        "scripts.extract_syntactic_features.subprocess.run",
        side_effect=error,
    ):
        result = extract_one(
            "QF_LIA/example.smt2",
            Path("/benchmarks"),
            Path("/bin/klhm"),
            timeout=5.0,
        )

    assert result.failed
    assert len(result.feature_values) == len(FIXED_COLUMNS) + len(SYMBOLS)
    assert not any(result.feature_values)


def test_paths_from_meta_rejects_duplicates(tmp_path):
    meta = tmp_path / "QF_LIA.json"
    meta.write_text(
        json.dumps(
            [
                {"smtlib_path": "QF_LIA/example.smt2"},
                {"smtlib_path": "QF_LIA/example.smt2"},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        paths_from_meta(meta)


def test_write_logic_tables_matches_experiment_layout(tmp_path):
    values = parse_klhm_output(klhm_payload())
    rows = [
        Extraction(
            relative_path="QF_LIA/example.smt2",
            elapsed_seconds=0.125,
            failed=False,
            feature_values=values,
        )
    ]

    write_logic_tables(tmp_path, rows)

    with (tmp_path / "features.csv").open(newline="", encoding="utf-8") as handle:
        feature_rows = list(csv.reader(handle))
    with (tmp_path / "extraction_times.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        timing_rows = list(csv.reader(handle))

    assert feature_rows[0] == ["path", *FIXED_COLUMNS, *SYMBOLS]
    assert feature_rows[1][0] == "QF_LIA/example.smt2"
    assert len(feature_rows[1]) == len(feature_rows[0])
    assert timing_rows == [
        ["path", "time_sec", "failed"],
        ["QF_LIA/example.smt2", "0.125", "0"],
    ]
