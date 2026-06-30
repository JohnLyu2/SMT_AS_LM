#!/usr/bin/env python3
"""
Graph + family-only description (Fusion-PWC): same as graph+text, but uses
  - GIN: --gin-features-base (default data/features/graph)
  - Text: data/features/desc_family_only/all-mpnet-base-v2
  - Lite+Text fallback rows: data/results/lite+family_only_desc/all-mpnet-base-v2

Forwards other arguments to experiment_graph_text.py.

--logic: pass one or more division names to run them in sequence (each gets its own
  output under data/results/graph+family_only_desc/all-mpnet-base-v2/<LOGIC>/ unless overridden).
  Omit --logic to run fusion's batch mode (all discoverable logics).

Fusion checkpoints: models/fusion_pwc_family_only_desc/<LOGIC>/seed<N>/.

Selector metrics are test-split only (see experiment_graph_text.py).

Examples:
  python scripts/experiment_graph_family_only_desc.py --logic BV
  python scripts/experiment_graph_family_only_desc.py --logic BV ABV QF_BV --filter
  python scripts/experiment_graph_family_only_desc.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DESC_FAMILY_ONLY = PROJECT_ROOT / "data" / "features" / "desc_family_only" / "all-mpnet-base-v2"
LITE_FAMILY_ONLY_DESC = PROJECT_ROOT / "data" / "results" / "lite+family_only_desc" / "all-mpnet-base-v2"
GIN_DEFAULT = PROJECT_ROOT / "data" / "features" / "graph"
OUTPUT_BASE = PROJECT_ROOT / "data" / "results" / "graph+family_only_desc" / "all-mpnet-base-v2"


def _strip_output_dir(argv: list[str]) -> tuple[list[str], str | None]:
    """Remove --output-dir VAL or --output-dir=VAL; return (rest, value or None)."""
    out: list[str] = []
    od_val: str | None = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--output-dir" and i + 1 < len(argv):
            od_val = argv[i + 1]
            i += 2
            continue
        if a.startswith("--output-dir="):
            od_val = a.split("=", 1)[1]
            i += 1
            continue
        out.append(a)
        i += 1
    return out, od_val


def _fusion_prefix() -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "experiment_graph_text.py"),
        "--desc-features-dir",
        str(DESC_FAMILY_ONLY),
        "--lite-text-dir",
        str(LITE_FAMILY_ONLY_DESC),
        "--gin-features-base",
        str(GIN_DEFAULT),
        "--save-models",
        "--fusion-models-subdir",
        "fusion_pwc_family_only_desc",
    ]


def main() -> int:
    wrapper = argparse.ArgumentParser(
        description="Graph + family-only-description fusion (delegates to experiment_graph_text.py).",
    )
    wrapper.add_argument(
        "--logic",
        nargs="*",
        metavar="LOGIC",
        help=(
            "One or more divisions, run in order. Omit this flag entirely to run all logics "
            "(fusion batch mode; requires output dir handling below)."
        ),
    )
    args, rest = wrapper.parse_known_args(sys.argv[1:])
    logics: list[str] | None = args.logic

    if logics is not None and len(logics) == 0:
        wrapper.error(
            "--logic requires at least one LOGIC, or omit --logic to run all divisions"
        )

    for p, label in (
        (DESC_FAMILY_ONLY, "family-only description features"),
        (LITE_FAMILY_ONLY_DESC, "lite+family-only-description results"),
        (GIN_DEFAULT, "GIN features"),
    ):
        if not p.is_dir():
            raise FileNotFoundError(f"Required directory for {label} not found: {p}")

    rest_wo_od, user_out = _strip_output_dir(rest)
    has_splits = any(
        x == "--splits-dir" or x.startswith("--splits-dir=") for x in rest_wo_od
    )

    # No --logic: same as before — batch all (inject default output-dir if needed)
    if logics is None:
        argv2 = list(rest_wo_od)
        has_output = user_out is not None
        if not has_output and not has_splits:
            argv2 = ["--output-dir", str(OUTPUT_BASE)] + argv2
        elif has_output:
            argv2 = ["--output-dir", user_out] + argv2
        cmd = _fusion_prefix() + argv2
        return subprocess.call(cmd, cwd=str(PROJECT_ROOT))

    # One or more explicit logics
    assert logics is not None and len(logics) >= 1

    def run_one(logic: str, output_dir: str) -> int:
        cmd = _fusion_prefix() + [
            "--logic",
            logic,
            "--output-dir",
            output_dir,
        ] + rest_wo_od
        return subprocess.call(cmd, cwd=str(PROJECT_ROOT))

    if len(logics) == 1:
        L = logics[0]
        if user_out is not None:
            out_dir = str(Path(user_out).resolve())
        else:
            out_dir = str(OUTPUT_BASE / L)
        return run_one(L, out_dir)

    # Multiple logics: optional --output-dir is a *parent*; each run uses <parent>/<LOGIC>
    base = Path(user_out).resolve() if user_out is not None else OUTPUT_BASE
    rc = 0
    for L in logics:
        out_dir = str(base / L)
        r = run_one(L, out_dir)
        if r != 0:
            rc = r
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
