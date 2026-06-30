#!/usr/bin/env python3
"""Evaluate a saved GIN-PWC selector."""

import json
import logging
from pathlib import Path

from smt_select.defaults import DEFAULT_BENCHMARK_ROOT
from smt_select.evaluation.evaluate import (
    as_evaluate,
    as_evaluate_parallel,
)
from smt_select.evaluation.metrics import compute_metrics, format_evaluation_short
from smt_select.models.graph.selector import GraphSelector
from smt_select.data.performance import MultiSolverDataset, parse_performance_json


def _load_gin_selector(model_dir: str, device: str | None = None):
    """Top-level loader for GIN (picklable for multiprocessing). device='cpu' in workers to avoid CUDA in forked processes."""
    with open(Path(model_dir) / "config.json") as f:
        config = json.load(f)
    if "num_solvers" in config:
        return GraphSelector.load(model_dir, device=device)
    raise ValueError(f"Invalid GIN-PWC config in {model_dir}: missing 'num_solvers'")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate GIN algorithm selector")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory with config.json, model.pt, vocab.json")
    parser.add_argument("--perf-json", type=str, required=True, help="Performance JSON (e.g. test.json)")
    parser.add_argument("--timeout", type=float, default=1200.0, help="PAR2 timeout in seconds")
    parser.add_argument("--benchmark-root", type=str, default=DEFAULT_BENCHMARK_ROOT, help="Root for relative instance paths")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV path for per-instance results")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers for evaluation; 1 = sequential")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    model_dir = Path(args.model_dir)
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    multi_perf_data = parse_performance_json(args.perf_json, args.timeout)
    if args.benchmark_root:
        root = Path(args.benchmark_root).resolve()
        if not root.is_dir():
            raise ValueError(f"--benchmark-root is not a directory: {root}")
        rebased = {str(root / p): multi_perf_data[p] for p in multi_perf_data.keys()}
        multi_perf_data = MultiSolverDataset(
            rebased,
            multi_perf_data.get_solver_id_dict(),
            multi_perf_data.get_timeout(),
        )
    instance_paths = list(multi_perf_data.keys())

    if args.jobs > 1:
        logging.info("Evaluating GIN-PWC with %d workers (CPU)", args.jobs)
        # Use CPU in workers to avoid CUDA init errors in forked processes.
        fallback_ids = config.get("fallback_solver_ids") or config.get("timeout_solver_ids") or []
        fallback_solver_id = fallback_ids[0] if fallback_ids else None
        result = as_evaluate_parallel(
            instance_paths,
            _load_gin_selector,
            (args.model_dir, "cpu"),
            multi_perf_data,
            n_workers=args.jobs,
            write_csv_path=args.output_csv,
            show_progress=True,
            fallback_solver_id=fallback_solver_id,
        )
    else:
        if "num_solvers" in config:
            selector = GraphSelector.load(args.model_dir)
            logging.info("Loaded GIN-PWC selector from %s", args.model_dir)
        else:
            raise ValueError("Invalid GIN-PWC config: missing 'num_solvers'")
        result = as_evaluate(
            selector,
            multi_perf_data,
            write_csv_path=args.output_csv,
            show_progress=True,
        )
    metrics = compute_metrics(result, multi_perf_data)
    print(format_evaluation_short(metrics))
    if args.output_csv:
        print(f"Wrote per-instance results to {args.output_csv}")
