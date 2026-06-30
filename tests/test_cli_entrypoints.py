"""Smoke tests for supported command-line entry points."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "script",
    [
        "scripts/encode_descriptions.py",
        "scripts/evaluate_gin.py",
        "scripts/evaluate_selector.py",
        "scripts/train_tabular.py",
        "scripts/train_setfit.py",
    ],
)
def test_cli_help(script):
    result = subprocess.run(
        [sys.executable, script, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
