"""Project-wide default paths. Override with CLI (e.g. --benchmark-root) or SMT_LIB_NONINC_ROOT when needed."""

import os
from pathlib import Path

# Repo default: smtlib/non-incremental (after scripts/download_smtlib.sh).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REPO_DEFAULT = _PROJECT_ROOT / "smtlib" / "non-incremental"

# Default root for relative instance paths in performance JSONs.
# Used as default for --benchmark-root in training and evaluation scripts.
# Override locally by setting SMT_LIB_NONINC_ROOT.
DEFAULT_BENCHMARK_ROOT = os.environ.get("SMT_LIB_NONINC_ROOT") or str(_REPO_DEFAULT)
