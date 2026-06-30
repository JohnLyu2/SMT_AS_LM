import json

import pytest

from scripts.latex.common import (
    collect_summary_logics,
    latex_escape,
    test_metric_mean_std as read_test_metric_mean_std,
)


def test_metric_reader_prefers_aggregated_values(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "aggregated": {
                    "test": {"gap_cls_par2_mean": 0.5, "gap_cls_par2_std": 0.1}
                }
            }
        ),
        encoding="utf-8",
    )
    assert read_test_metric_mean_std(path, "gap_cls_par2") == (0.5, 0.1)


def test_metric_reader_falls_back_to_seed_values(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "seeds": [
                    {"test_metrics": {"solved": 2}},
                    {"test_metrics": {"solved": 4}},
                ]
            }
        ),
        encoding="utf-8",
    )
    mean, std = read_test_metric_mean_std(path, "solved")
    assert mean == 3
    assert std == pytest.approx(1)


def test_latex_escape_and_logic_discovery(tmp_path):
    logic = tmp_path / "QF_BV"
    logic.mkdir()
    (logic / "summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "empty").mkdir()

    assert latex_escape("QF_BV & 10%") == r"QF\_BV \& 10\%"
    assert collect_summary_logics(tmp_path) == {"QF_BV"}
