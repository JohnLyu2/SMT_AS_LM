"""Selectors based on precomputed feature tables."""

from smt_select.models.tabular.selector import (
    PairwiseSVM,
    TabularSelector,
    train_tabular_selector,
)

__all__ = ["PairwiseSVM", "TabularSelector", "train_tabular_selector"]
