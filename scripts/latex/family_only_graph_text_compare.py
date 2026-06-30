#!/usr/bin/env python3
"""
Generate doc/cp26/family_only_graph_text_compare.tex comparing PAR-2 gap closed (%)
for Graph vs. Graph + Text (family-only and full desc).
"""

from pathlib import Path

try:
    from scripts.latex.common import (
        collect_summary_logics as collect_logics_from_dir,
        latex_escape,
        test_metric_mean_std,
    )
except ModuleNotFoundError:
    from common import (
        collect_summary_logics as collect_logics_from_dir,
        latex_escape,
        test_metric_mean_std,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_ROOT = PROJECT_ROOT / "data" / "results" / "graph"
GRAPH_TEXT_ROOT = PROJECT_ROOT / "data" / "results" / "graph+text" / "all-mpnet-base-v2"
GRAPH_FAMILY_ONLY_ROOT = PROJECT_ROOT / "data" / "results" / "graph+family_only_desc" / "all-mpnet-base-v2"

# Column order: Graph | Graph+Text family-only | Graph+Text full desc (result paths)
LABELS = ["graph", "family_only_desc", "graph_text"]

TEX_PATH = PROJECT_ROOT / "doc" / "cp26" / "family_only_graph_text_compare.tex"


def get_test_gap_par2(summary_path: Path) -> tuple[float | None, float | None]:
    return test_metric_mean_std(summary_path, "gap_cls_par2")


def format_cell(mean: float | None) -> str:
    if mean is None:
        return "---"
    return f"{mean * 100:.2f}"


def main() -> None:
    graph_root = GRAPH_ROOT.resolve()
    graph_text_root = GRAPH_TEXT_ROOT.resolve()
    graph_family_only_root = GRAPH_FAMILY_ONLY_ROOT.resolve()
    for d in (graph_root, graph_text_root, graph_family_only_root):
        if not d.is_dir():
            raise FileNotFoundError(f"Result directory not found: {d}")

    all_logics: set[str] = set()
    for name in collect_logics_from_dir(graph_root):
        all_logics.add(name)
    for name in collect_logics_from_dir(graph_text_root):
        all_logics.add(name)
    for name in collect_logics_from_dir(graph_family_only_root):
        all_logics.add(name)
    logics = sorted(all_logics)

    data: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    for logic in logics:
        data[(logic, "graph")] = get_test_gap_par2(graph_root / logic / "summary.json")
        data[(logic, "graph_text")] = get_test_gap_par2(
            graph_text_root / logic / "summary.json"
        )
        data[(logic, "family_only_desc")] = get_test_gap_par2(
            graph_family_only_root / logic / "summary.json"
        )

    col_spec = "l" + "c" * len(LABELS)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & Without Description & Family-only Description & Full Description \\\\",
        "\\midrule",
    ]

    for logic in logics:
        row_vals: dict[str, float] = {}
        for label in LABELS:
            mean, _ = data.get((logic, label), (None, None))
            if mean is not None:
                row_vals[label] = mean
        best_label: str | None = None
        if row_vals:
            m = max(row_vals.values())
            for label in LABELS:
                v = row_vals.get(label)
                if v is not None and v == m:
                    best_label = label
                    break

        cells = [latex_escape(logic)]
        for label in LABELS:
            mean, _ = data.get((logic, label), (None, None))
            cell = format_cell(mean)
            if label == best_label:
                cells.append("\\textbf{" + cell + "}")
            else:
                cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Performance comparison of \\smtselect{} (Graph) under different natural-language "
        "description modes, measured by PAR-2 SBS--VBS gap closed (\\%) on the test set "
        "(mean over five random train--test splits).}",
        "\\label{tab:family_only_desc}",
        "\\end{table}",
    ])

    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {TEX_PATH}")


if __name__ == "__main__":
    main()
