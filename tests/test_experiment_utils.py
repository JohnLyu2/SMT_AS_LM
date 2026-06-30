"""Tests for shared multi-split experiment orchestration."""

from pathlib import Path

import numpy as np
import pytest

from smt_select.evaluation.experiment_utils import (
    aggregate_gap_metrics,
    discover_seed_dirs,
    select_seed_dirs,
    to_json_value,
)


def make_split(root: Path, seed: int, *, complete: bool = True) -> Path:
    split = root / f"seed{seed}"
    split.mkdir()
    (split / "train.json").write_text("[]", encoding="utf-8")
    if complete:
        (split / "test.json").write_text("[]", encoding="utf-8")
    return split


def test_discover_seed_dirs_is_numeric_and_requires_both_files(tmp_path):
    make_split(tmp_path, 20)
    make_split(tmp_path, 3)
    make_split(tmp_path, 10, complete=False)
    (tmp_path / "notes").mkdir()

    assert [seed for seed, _ in discover_seed_dirs(tmp_path)] == [3, 20]


def test_select_seed_dirs_rejects_missing_seed(tmp_path):
    entries = [(0, tmp_path / "seed0"), (10, tmp_path / "seed10")]

    assert [seed for seed, _ in select_seed_dirs(entries, [10])] == [10]
    with pytest.raises(ValueError, match=r"\[20\]"):
        select_seed_dirs(entries, [20])


def test_aggregate_gap_metrics_supports_requested_splits():
    rows = [
        {
            "test_metrics": {"gap_cls_solved": 0.2, "gap_cls_par2": 0.4},
            "train_metrics": {"gap_cls_solved": 0.8, "gap_cls_par2": 0.6},
        },
        {
            "test_metrics": {"gap_cls_solved": 0.4, "gap_cls_par2": 0.8},
            "train_metrics": {"gap_cls_solved": 1.0, "gap_cls_par2": 1.0},
        },
    ]

    aggregated = aggregate_gap_metrics(rows, ("train", "test"))

    assert aggregated["test"]["gap_cls_solved_mean"] == pytest.approx(0.3)
    assert aggregated["test"]["gap_cls_par2_std"] == pytest.approx(0.2)
    assert aggregated["train"]["gap_cls_solved_mean"] == pytest.approx(0.9)


def test_to_json_value_converts_numpy_recursively():
    converted = to_json_value(
        {"integer": np.int64(2), "array": np.array([1.5]), "items": (np.float32(3),)}
    )

    assert converted == {"integer": 2, "array": [1.5], "items": [3.0]}
