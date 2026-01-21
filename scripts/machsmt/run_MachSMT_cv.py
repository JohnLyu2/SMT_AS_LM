#!/usr/bin/env python3
"""
Cross-validation script for MachSMT algorithm selection.

Performs k-fold cross-validation using pre-split fold files:
1. For each fold, using the fold file as the test set
2. Training MachSMT on all other folds combined (train set = all folds - test fold)
3. Evaluating on the held-out fold
4. Aggregating results across all folds

Output structure matches existing CV results format:
- {output_dir}/summary.json - aggregated metrics
- {output_dir}/test/{fold_num}.csv - per-fold test predictions
- {output_dir}/train/{fold_num}.csv - per-fold train predictions
"""

import os
import sys
import csv
import json
import argparse
import tempfile
import shutil
from pathlib import Path
import numpy as np


# Parse config argument BEFORE importing MachSMT (to avoid MachSMT's argparse taking over)
def parse_config_argument():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file')
    args, remaining = parser.parse_known_args()
    sys.argv[1:] = remaining
    return args


_config_args = parse_config_argument()
CONFIG_PATH = _config_args.config


# Add MachSMT directory to Python path to use local version
machsmt_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    'MachSMT'
)
sys.path.insert(0, machsmt_dir)

from machsmt import MachSMT, Benchmark
from machsmt import args as machsmt_args
from machsmt.database import DataBase


def load_config(config_file):
    """Load configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def parse_fold_csv(fold_file, timeout):
    """
    Parse a fold CSV file and return performance data.

    CSV format (from foiks26 folds):
    path,Solver1,,Solver2,,Solver3,
    ,solved,runtime,solved,runtime,solved,runtime
    benchmark_path,0,1.23,1,0.45,...

    Returns:
        dict: {benchmark_path: {solver_name: {'solved': int, 'runtime': float, 'score': float}}}
        list: List of solver names
    """
    results = {}
    solver_names = []

    with open(fold_file, 'r') as f:
        reader = csv.reader(f)

        # First row: path,Solver1,,Solver2,,Solver3,
        header_row = next(reader)
        # Extract solver names (every other column starting from index 1)
        solver_names = [header_row[i] for i in range(1, len(header_row), 2) if header_row[i]]

        # Second row: ,solved,runtime,solved,runtime,solved,runtime
        next(reader)  # Skip this row

        # Data rows
        for row in reader:
            if not row or not row[0]:
                continue

            benchmark_path = row[0]
            results[benchmark_path] = {}

            for i, solver in enumerate(solver_names):
                col_base = 1 + i * 2  # solved column index
                if col_base + 1 < len(row):
                    solved = int(row[col_base]) if row[col_base] else 0
                    runtime = float(row[col_base + 1]) if row[col_base + 1] else timeout

                    # Calculate PAR-2 score
                    if solved:
                        score = runtime
                    else:
                        score = timeout * 2

                    results[benchmark_path][solver] = {
                        'solved': solved,
                        'runtime': runtime,
                        'score': score
                    }

    return results, solver_names


def combine_folds_for_training(fold_files, exclude_fold_idx, timeout):
    """
    Combine all fold files except the excluded one for training.

    Returns a temporary CSV file path in MachSMT's expected LONG format:
    benchmark,solver,result,runtime,score
    """
    all_results = {}
    solver_names = None

    for i, fold_file in enumerate(fold_files):
        if i == exclude_fold_idx:
            continue

        fold_results, fold_solvers = parse_fold_csv(fold_file, timeout)
        all_results.update(fold_results)

        if solver_names is None:
            solver_names = fold_solvers

    # Create temporary CSV file for MachSMT training in LONG format
    temp_fd, temp_path = tempfile.mkstemp(suffix='.csv')
    os.close(temp_fd)

    with open(temp_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header in MachSMT's expected format
        writer.writerow(['benchmark', 'solver', 'result', 'runtime', 'score'])

        # Write data rows - one row per (benchmark, solver) pair
        for benchmark_path, solver_data in all_results.items():
            for solver in solver_names:
                if solver in solver_data:
                    data = solver_data[solver]
                    writer.writerow([
                        benchmark_path,
                        solver,
                        data['solved'],  # result: 0 or 1
                        data['runtime'],
                        data['score']    # PAR-2 score
                    ])
                else:
                    # Solver not in data - treat as unsolved
                    writer.writerow([
                        benchmark_path,
                        solver,
                        0,
                        timeout,
                        timeout * 2
                    ])

    return temp_path, all_results, solver_names


def get_best_solver_performance(results, solver_names, timeout):
    """
    Get the Single Best Solver (SBS) performance.
    Returns the solver with highest solve rate.
    """
    solver_stats = {solver: {'solved': 0, 'total_par2': 0} for solver in solver_names}

    for benchmark_path, solver_data in results.items():
        for solver, data in solver_data.items():
            if solver in solver_stats:
                solver_stats[solver]['solved'] += data['solved']
                solver_stats[solver]['total_par2'] += data['score']

    # Find best solver by solve count
    best_solver = max(solver_stats.keys(), key=lambda s: solver_stats[s]['solved'])

    return best_solver, solver_stats[best_solver]


def get_virtual_best_performance(results, timeout):
    """
    Get the Virtual Best Solver (VBS) performance.
    For each benchmark, use the best solver.
    """
    vbs_solved = 0
    vbs_total_par2 = 0

    for benchmark_path, solver_data in results.items():
        # Find best solver for this benchmark
        best_score = float('inf')
        best_solved = 0

        for solver, data in solver_data.items():
            if data['score'] < best_score:
                best_score = data['score']
                best_solved = data['solved']

        vbs_solved += best_solved
        vbs_total_par2 += best_score

    return vbs_solved, vbs_total_par2


def evaluate_machsmt_predictions(prediction_model, results, benchmark_prefix, timeout, selector):
    """
    Evaluate MachSMT predictions on a set of benchmarks.

    Returns:
        list: CSV data rows with predictions
        dict: Metrics (solved, total_par2, etc.)
    """
    csv_data = []
    total_solved = 0
    total_par2 = 0
    parse_failures = 0

    # Change to benchmark directory for MachSMT
    original_cwd = os.getcwd()
    os.chdir(benchmark_prefix)

    try:
        for benchmark_path, solver_data in results.items():
            full_path = os.path.join(benchmark_prefix, benchmark_path)

            if not os.path.exists(full_path):
                # Benchmark file not found - treat as parse failure
                parse_failures += 1
                csv_data.append({
                    'benchmark': benchmark_path,
                    'selected': 'FILE_NOT_FOUND',
                    'solved': 0,
                    'runtime': timeout
                })
                total_par2 += timeout * 2
                continue

            try:
                benchmark = Benchmark(full_path)
                benchmark.parse()

                # Get MachSMT prediction
                predictions, _ = prediction_model.predict([benchmark], include_predictions=True, selector=selector)
                selected_solver = predictions[0].get_name()

                # Get actual result for selected solver
                if selected_solver in solver_data:
                    result_data = solver_data[selected_solver]
                    solved = result_data['solved']
                    runtime = result_data['runtime']
                    score = result_data['score']
                else:
                    # Solver not in results - treat as unsolved
                    solved = 0
                    runtime = timeout
                    score = timeout * 2

                total_solved += solved
                total_par2 += score

                csv_data.append({
                    'benchmark': benchmark_path,
                    'selected': selected_solver,
                    'solved': solved,
                    'runtime': runtime
                })

            except Exception as e:
                # Parse failure
                parse_failures += 1
                csv_data.append({
                    'benchmark': benchmark_path,
                    'selected': 'PARSE_FAILED',
                    'solved': 0,
                    'runtime': timeout
                })
                total_par2 += timeout * 2

    finally:
        os.chdir(original_cwd)

    total_instances = len(results)

    metrics = {
        'total_instances': total_instances,
        'solved': total_solved,
        'solve_rate': (total_solved / total_instances * 100) if total_instances > 0 else 0,
        'total_par2': total_par2,
        'avg_par2': total_par2 / total_instances if total_instances > 0 else 0,
        'parse_failures': parse_failures
    }

    return csv_data, metrics


def compute_full_metrics(as_metrics, results, solver_names, timeout):
    """
    Compute full metrics including SBS, VBS, and gap_cls.
    """
    # Get SBS performance
    sbs_solver, sbs_stats = get_best_solver_performance(results, solver_names, timeout)
    sbs_solved = sbs_stats['solved']
    sbs_total_par2 = sbs_stats['total_par2']
    total_instances = as_metrics['total_instances']

    # Get VBS performance
    vbs_solved, vbs_total_par2 = get_virtual_best_performance(results, timeout)

    # Calculate rates and averages
    sbs_solve_rate = (sbs_solved / total_instances * 100) if total_instances > 0 else 0
    sbs_avg_par2 = sbs_total_par2 / total_instances if total_instances > 0 else 0

    vbs_solve_rate = (vbs_solved / total_instances * 100) if total_instances > 0 else 0
    vbs_avg_par2 = vbs_total_par2 / total_instances if total_instances > 0 else 0

    # Assert that SBS and VBS are not the same
    assert vbs_solved != sbs_solved, (
        f"SBS and VBS have the same solved count: {sbs_solved}"
    )
    assert vbs_avg_par2 != sbs_avg_par2, (
        f"SBS and VBS have the same avg_par2: {sbs_avg_par2}"
    )

    # Calculate gap_cls metrics: (as - sbs) / (vbs - sbs)
    gap_cls_solved = (as_metrics['solved'] - sbs_solved) / (vbs_solved - sbs_solved)
    gap_cls_par2 = (as_metrics['avg_par2'] - sbs_avg_par2) / (vbs_avg_par2 - sbs_avg_par2)

    return {
        'total_instances': total_instances,
        'solved': as_metrics['solved'],
        'solve_rate': as_metrics['solve_rate'],
        'avg_par2': as_metrics['avg_par2'],
        'sbs_name': sbs_solver,
        'sbs_solved': sbs_solved,
        'sbs_solve_rate': sbs_solve_rate,
        'sbs_avg_par2': sbs_avg_par2,
        'vbs_solved': vbs_solved,
        'vbs_solve_rate': vbs_solve_rate,
        'vbs_avg_par2': vbs_avg_par2,
        'gap_cls_solved': gap_cls_solved,
        'gap_cls_par2': gap_cls_par2,
    }


def write_csv_results(csv_data, output_path):
    """Write prediction results to CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['benchmark', 'selected', 'solved', 'runtime'])
        writer.writeheader()
        writer.writerows(csv_data)


def cross_validate_machsmt(folds_dir, benchmark_prefix, output_dir, timeout, selector):
    """
    Perform k-fold cross-validation for MachSMT.

    Args:
        folds_dir: Directory containing fold CSV files (0.csv, 1.csv, ...)
        benchmark_prefix: Root directory where benchmark files are stored
        output_dir: Directory to save results
        timeout: Timeout value in seconds
        selector: MachSMT selector (e.g., 'EHM')

    Returns:
        dict: Cross-validation results
    """
    folds_dir = Path(folds_dir)
    output_dir = Path(output_dir)

    # Find all fold files
    fold_files = sorted(folds_dir.glob("*.csv"), key=lambda x: int(x.stem))
    n_splits = len(fold_files)

    if n_splits == 0:
        raise ValueError(f"No fold CSV files found in {folds_dir}")

    print(f"\n{'=' * 70}")
    print(f"MachSMT Cross-Validation")
    print(f"{'=' * 70}")
    print(f"Folds directory: {folds_dir}")
    print(f"Number of folds: {n_splits}")
    print(f"Benchmark prefix: {benchmark_prefix}")
    print(f"Timeout: {timeout}s")
    print(f"Selector: {selector}")
    print(f"Output directory: {output_dir}")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir = output_dir / "test"
    train_output_dir = output_dir / "train"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    train_output_dir.mkdir(parents=True, exist_ok=True)

    # Configure MachSMT
    machsmt_args.original_features = True
    machsmt_args.description = False
    machsmt_args.description_family = False
    machsmt_args.feature_timeout = 60
    machsmt_args.cores = 16

    # Storage for per-fold results
    fold_results = []

    # Get total instances from all folds
    all_instances = set()
    for fold_file in fold_files:
        fold_data, _ = parse_fold_csv(fold_file, timeout)
        all_instances.update(fold_data.keys())
    n_instances = len(all_instances)

    print(f"Total instances: {n_instances}")

    # Iterate over folds
    for fold_num, fold_file in enumerate(fold_files):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_num + 1}/{n_splits} ({fold_file.name})")
        print(f"{'=' * 60}")

        # Load test fold
        test_results, solver_names = parse_fold_csv(fold_file, timeout)
        test_size = len(test_results)
        print(f"Test instances: {test_size}")

        # Combine all other folds for training
        train_csv_path, train_results, _ = combine_folds_for_training(
            fold_files, fold_num, timeout
        )
        train_size = len(train_results)
        print(f"Train instances: {train_size}")

        try:
            # Train MachSMT
            print("Training MachSMT model...")

            # Create temporary results directory for MachSMT
            temp_results_dir = tempfile.mkdtemp()
            machsmt_args.results = temp_results_dir

            # Change to benchmark directory and train
            original_cwd = os.getcwd()
            os.chdir(benchmark_prefix)

            db = DataBase(build_on_init=False)
            db.build([train_csv_path])

            machsmt = MachSMT(db, train_on_init=True)

            os.chdir(original_cwd)
            print("Model training completed")

            # Evaluate on train set
            print("Evaluating on train set...")
            train_csv_data, train_metrics = evaluate_machsmt_predictions(
                machsmt, train_results, benchmark_prefix, timeout, selector
            )
            train_full_metrics = compute_full_metrics(
                train_metrics, train_results, solver_names, timeout
            )

            # Evaluate on test set
            print("Evaluating on test set...")
            test_csv_data, test_metrics = evaluate_machsmt_predictions(
                machsmt, test_results, benchmark_prefix, timeout, selector
            )
            test_full_metrics = compute_full_metrics(
                test_metrics, test_results, solver_names, timeout
            )

            # Write CSV results
            write_csv_results(train_csv_data, train_output_dir / f"{fold_num}.csv")
            write_csv_results(test_csv_data, test_output_dir / f"{fold_num}.csv")

            # Store fold results
            fold_result = {
                "fold": fold_num + 1,
                "train_size": train_size,
                "test_size": test_size,
                "train_metrics": train_full_metrics,
                "test_metrics": test_full_metrics,
            }
            fold_results.append(fold_result)

            # Log fold results
            print(f"\nFold {fold_num + 1} Results:")
            print("  Train Set:")
            print(f"    Solved: {train_full_metrics['solved']}/{train_full_metrics['total_instances']}")
            print(f"    Solve rate: {train_full_metrics['solve_rate']:.2f}%")
            print(f"    Average PAR-2: {train_full_metrics['avg_par2']:.2f}")
            print("  Test Set:")
            print(f"    Solved: {test_full_metrics['solved']}/{test_full_metrics['total_instances']}")
            print(f"    Solve rate: {test_full_metrics['solve_rate']:.2f}%")
            print(f"    Average PAR-2: {test_full_metrics['avg_par2']:.2f}")

            # Clean up temporary files
            shutil.rmtree(temp_results_dir, ignore_errors=True)

        finally:
            # Clean up training CSV
            if os.path.exists(train_csv_path):
                os.remove(train_csv_path)

    # Aggregate results across folds
    test_metrics_list = [fr["test_metrics"] for fr in fold_results]
    train_metrics_list = [fr["train_metrics"] for fr in fold_results]

    aggregated = {
        "test_solve_rate_mean": np.mean([m["solve_rate"] for m in test_metrics_list]),
        "test_solve_rate_std": np.std([m["solve_rate"] for m in test_metrics_list]),
        "test_avg_par2_mean": np.mean([m["avg_par2"] for m in test_metrics_list]),
        "test_avg_par2_std": np.std([m["avg_par2"] for m in test_metrics_list]),
        "test_sbs_solve_rate_mean": np.mean([m["sbs_solve_rate"] for m in test_metrics_list]),
        "test_sbs_solve_rate_std": np.std([m["sbs_solve_rate"] for m in test_metrics_list]),
        "test_sbs_avg_par2_mean": np.mean([m["sbs_avg_par2"] for m in test_metrics_list]),
        "test_sbs_avg_par2_std": np.std([m["sbs_avg_par2"] for m in test_metrics_list]),
        "test_vbs_solve_rate_mean": np.mean([m["vbs_solve_rate"] for m in test_metrics_list]),
        "test_vbs_solve_rate_std": np.std([m["vbs_solve_rate"] for m in test_metrics_list]),
        "test_vbs_avg_par2_mean": np.mean([m["vbs_avg_par2"] for m in test_metrics_list]),
        "test_vbs_avg_par2_std": np.std([m["vbs_avg_par2"] for m in test_metrics_list]),
        "test_gap_cls_solved_mean": np.mean([m["gap_cls_solved"] for m in test_metrics_list]),
        "test_gap_cls_solved_std": np.std([m["gap_cls_solved"] for m in test_metrics_list]),
        "test_gap_cls_par2_mean": np.mean([m["gap_cls_par2"] for m in test_metrics_list]),
        "test_gap_cls_par2_std": np.std([m["gap_cls_par2"] for m in test_metrics_list]),
        "train_solve_rate_mean": np.mean([m["solve_rate"] for m in train_metrics_list]),
        "train_solve_rate_std": np.std([m["solve_rate"] for m in train_metrics_list]),
        "train_avg_par2_mean": np.mean([m["avg_par2"] for m in train_metrics_list]),
        "train_avg_par2_std": np.std([m["avg_par2"] for m in train_metrics_list]),
        "train_sbs_solve_rate_mean": np.mean([m["sbs_solve_rate"] for m in train_metrics_list]),
        "train_sbs_solve_rate_std": np.std([m["sbs_solve_rate"] for m in train_metrics_list]),
        "train_sbs_avg_par2_mean": np.mean([m["sbs_avg_par2"] for m in train_metrics_list]),
        "train_sbs_avg_par2_std": np.std([m["sbs_avg_par2"] for m in train_metrics_list]),
        "train_vbs_solve_rate_mean": np.mean([m["vbs_solve_rate"] for m in train_metrics_list]),
        "train_vbs_solve_rate_std": np.std([m["vbs_solve_rate"] for m in train_metrics_list]),
        "train_vbs_avg_par2_mean": np.mean([m["vbs_avg_par2"] for m in train_metrics_list]),
        "train_vbs_avg_par2_std": np.std([m["vbs_avg_par2"] for m in train_metrics_list]),
        "train_gap_cls_solved_mean": np.mean([m["gap_cls_solved"] for m in train_metrics_list]),
        "train_gap_cls_solved_std": np.std([m["gap_cls_solved"] for m in train_metrics_list]),
        "train_gap_cls_par2_mean": np.mean([m["gap_cls_par2"] for m in train_metrics_list]),
        "train_gap_cls_par2_std": np.std([m["gap_cls_par2"] for m in train_metrics_list]),
    }

    results = {
        "n_splits": n_splits,
        "n_instances": n_instances,
        "model_type": "MachSMT",
        "folds_dir": str(folds_dir),
        "selector": selector,
        "timeout": timeout,
        "folds": fold_results,
        "aggregated": aggregated,
    }

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    json_results = convert_to_json_serializable(results)

    # Save summary.json
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nSaved summary to {summary_path}")

    # Print aggregated results
    print("\n" + "=" * 60)
    print("Cross-Validation Results Summary")
    print("=" * 60)
    print(f"Model type: MachSMT")
    print(f"Number of folds: {n_splits}")
    print(f"Total instances: {n_instances}")
    print("")
    agg = aggregated

    print("Train Set Performance:")
    print("  Algorithm Selection:")
    print(f"    Solve rate: {agg['train_solve_rate_mean']:.2f}% ± {agg['train_solve_rate_std']:.2f}%")
    print(f"    Average PAR-2: {agg['train_avg_par2_mean']:.2f} ± {agg['train_avg_par2_std']:.2f}")
    print("  SBS (Single Best Solver):")
    print(f"    Solve rate: {agg['train_sbs_solve_rate_mean']:.2f}% ± {agg['train_sbs_solve_rate_std']:.2f}%")
    print(f"    Average PAR-2: {agg['train_sbs_avg_par2_mean']:.2f} ± {agg['train_sbs_avg_par2_std']:.2f}")
    print("  VBS (Virtual Best Solver):")
    print(f"    Solve rate: {agg['train_vbs_solve_rate_mean']:.2f}% ± {agg['train_vbs_solve_rate_std']:.2f}%")
    print(f"    Average PAR-2: {agg['train_vbs_avg_par2_mean']:.2f} ± {agg['train_vbs_avg_par2_std']:.2f}")
    print("")
    print("Test Set Performance:")
    print("  Algorithm Selection:")
    print(f"    Solve rate: {agg['test_solve_rate_mean']:.2f}% ± {agg['test_solve_rate_std']:.2f}%")
    print(f"    Average PAR-2: {agg['test_avg_par2_mean']:.2f} ± {agg['test_avg_par2_std']:.2f}")
    print("  SBS (Single Best Solver):")
    print(f"    Solve rate: {agg['test_sbs_solve_rate_mean']:.2f}% ± {agg['test_sbs_solve_rate_std']:.2f}%")
    print(f"    Average PAR-2: {agg['test_sbs_avg_par2_mean']:.2f} ± {agg['test_sbs_avg_par2_std']:.2f}")
    print("  VBS (Virtual Best Solver):")
    print(f"    Solve rate: {agg['test_vbs_solve_rate_mean']:.2f}% ± {agg['test_vbs_solve_rate_std']:.2f}%")
    print(f"    Average PAR-2: {agg['test_vbs_avg_par2_mean']:.2f} ± {agg['test_vbs_avg_par2_std']:.2f}")
    print("  Gap Closed:")
    print(f"    Solved: {agg['test_gap_cls_solved_mean']:.4f} ± {agg['test_gap_cls_solved_std']:.4f}")
    print(f"    PAR-2: {agg['test_gap_cls_par2_mean']:.4f} ± {agg['test_gap_cls_par2_std']:.4f}")
    print("=" * 60)

    return results


def main():
    # Load configuration (CONFIG_PATH was parsed before MachSMT import)
    config = load_config(CONFIG_PATH)

    folds_dir = config["paths"]["folds_dir"]
    benchmark_prefix = config["paths"]["benchmark_prefix"]
    output_dir = config["paths"]["output_dir"]
    timeout = int(config["settings"].get("timeout", 1200))
    selector = config["settings"].get("selector", "EHM")

    # Run cross-validation
    cross_validate_machsmt(
        folds_dir=folds_dir,
        benchmark_prefix=benchmark_prefix,
        output_dir=output_dir,
        timeout=timeout,
        selector=selector,
    )


if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    main()
