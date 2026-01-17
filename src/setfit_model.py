import argparse
import json
from pathlib import Path

from setfit import SetFitModel

from .parser import parse_performance_csv

model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def create_setfit_data(
    perf_csv_path: str,
    desc_json_path: str,
    timeout: float = 1200.0,
) -> dict[str, list]:
    """
    Create SetFit training data from performance CSV and description JSON.

    Args:
        perf_csv_path: Path to performance CSV file.
        desc_json_path: Path to description JSON file.
        timeout: Timeout value in seconds.
    Returns:
        Dict with keys: "texts", "labels", "paths".
    """
    desc_map = _load_description_map(desc_json_path)
    multi_perf_data = parse_performance_csv(perf_csv_path, timeout)

    texts: list[str] = []
    labels: list[str] = []
    paths: list[str] = []

    for path in multi_perf_data.keys():
        description = desc_map.get(path)
        if not description or not description.strip():
            raise AssertionError(f"Missing description for benchmark: {path}")

        solver_name = multi_perf_data.get_best_solver_for_instance(path)
        if solver_name is None:
            continue

        label = solver_name

        texts.append(description.strip())
        labels.append(label)
        paths.append(path)

    return {"texts": texts, "labels": labels, "paths": paths}


def _load_description_map(desc_json_path: str) -> dict[str, str]:
    json_path = Path(desc_json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    desc_map: dict[str, str] = {}
    for benchmark in benchmarks:
        smtlib_path = benchmark.get("smtlib_path", "")
        if not smtlib_path:
            continue

        description = benchmark.get("description", "")
        if not description or not description.strip():
            raise AssertionError(f"Missing description for benchmark: {smtlib_path}")

        desc_map[smtlib_path] = description.strip()

    return desc_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create SetFit data from performance CSV and descriptions JSON."
    )
    parser.add_argument(
        "--perf-csv",
        required=True,
        help="Path to performance CSV.",
    )
    parser.add_argument("--desc-json", required=True, help="Path to descriptions JSON.")
    parser.add_argument(
        "--timeout",
        default=1200.0,
        type=float,
        help="Timeout value used in performance data (seconds).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write SetFit data as JSON.",
    )

    args = parser.parse_args()

    data = create_setfit_data(args.perf_csv, args.desc_json, args.timeout)
    print(f"Total samples: {len(data['texts'])}")
    print(f"Unique labels: {len(set(data['labels']))}")

    if args.output_json:
        output_path = Path(args.output_json)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Wrote SetFit data to: {output_path}")


if __name__ == "__main__":
    main()
