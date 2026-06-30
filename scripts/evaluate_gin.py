#!/usr/bin/env python3
"""Thin runner for GIN evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smt_select.evaluation.evaluate_gin import main

if __name__ == "__main__":
    main()
