#!/usr/bin/env python3
"""
Cross-validation script for algorithm selection models.

Performs k-fold cross-validation by:
1. Splitting the dataset into k folds
2. Training a model on k-1 folds
3. Evaluating on the held-out fold
4. Aggregating results across all folds
"""

import csv
import json
import argparse
import logging
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold

from .performance import MultiSolverDataset
from .parser import parse_performance_csv
from .pwc import train_pwc, PwcModel
from .evaluate import as_evaluate


def create_subset_dataset(
    full_dataset: MultiSolverDataset, instance_paths: list[str]
) -> MultiSolverDataset:
    """
    Create a subset of MultiSolverDataset containing only the specified instance paths.

    Args:
        full_dataset: The full MultiSolverDataset
        instance_paths: List of instance paths to include in the subset

    Returns:
        A new MultiSolverDataset containing only the specified instances
    """
    subset_perf_dict = {}
    for path in instance_paths:
        if path in full_dataset:
            subset_perf_dict[path] = full_dataset[path]

    return MultiSolverDataset(
        subset_perf_dict,
        full_dataset.get_solver_id_dict(),
        full_dataset.get_timeout(),
    )


def compute_metrics(result_dataset, multi_perf_data):
    """
    Compute evaluation metrics for a result dataset.

    Args:
        result_dataset: SingleSolverDataset with algorithm selection results
        multi_perf_data: MultiSolverDataset for comparison metrics

    Returns:
        Dictionary of metrics
    """
    total_count = len(result_dataset)
    solved_count = result_dataset.get_solved_count()
    solve_rate = (solved_count / total_count * 100) if total_count > 0 else 0.0

    # Calculate average PAR-2
    total_par2 = sum(result_dataset.get_par2(path) for path in result_dataset.keys())
    avg_par2 = total_par2 / total_count if total_count > 0 else 0.0

    # Get best single solver for comparison
    best_solver_dataset = multi_perf_data.get_best_solver_dataset()
    best_solver_solved = best_solver_dataset.get_solved_count()
    best_solver_solve_rate = (
        (best_solver_solved / total_count * 100) if total_count > 0 else 0.0
    )
    total_par2_best = sum(
        best_solver_dataset.get_par2(path) for path in best_solver_dataset.keys()
    )
    avg_par2_best = total_par2_best / total_count if total_count > 0 else 0.0

    # Get virtual best solver for comparison
    virtual_best_dataset = multi_perf_data.get_virtual_best_solver_dataset()
    virtual_best_solved = virtual_best_dataset.get_solved_count()
    virtual_best_solve_rate = (
        (virtual_best_solved / total_count * 100) if total_count > 0 else 0.0
    )
    total_par2_virtual_best = sum(
        virtual_best_dataset.get_par2(path) for path in virtual_best_dataset.keys()
    )
    avg_par2_virtual_best = (
        total_par2_virtual_best / total_count if total_count > 0 else 0.0
    )

    return {
        "total_instances": total_count,
        "solved": solved_count,
        "solve_rate": solve_rate,
        "avg_par2": avg_par2,
        "best_solver_name": best_solver_dataset.get_solver_name(),
        "best_solver_solved": best_solver_solved,
        "best_solver_solve_rate": best_solver_solve_rate,
        "best_solver_avg_par2": avg_par2_best,
        "virtual_best_solved": virtual_best_solved,
        "virtual_best_solve_rate": virtual_best_solve_rate,
        "virtual_best_avg_par2": avg_par2_virtual_best,
    }


def cross_validate(
    multi_perf_data: MultiSolverDataset,
    feature_csv_path: str,
    n_splits: int = 5,
    xg_flag: bool = False,
    random_state: int = 42,
    save_models: bool = False,
    output_dir: Path = None,
    timeout: float = 1200.0,
):
    """
    Perform k-fold cross-validation on algorithm selection model.

    Args:
        multi_perf_data: MultiSolverDataset containing all performance data
        feature_csv_path: Path to features CSV file
        n_splits: Number of folds (default: 5)
        xg_flag: Whether to use XGBoost (default: False, uses SVM)
        random_state: Random seed for reproducibility
        save_models: Whether to save models for each fold
        output_dir: Directory to save results and models (optional)
        timeout: Timeout value in seconds

    Returns:
        Dictionary containing per-fold and aggregated results
    """
    # Get all instance paths
    instance_paths = list(multi_perf_data.keys())
    n_instances = len(instance_paths)

    if n_instances < n_splits:
        raise ValueError(
            f"Not enough instances ({n_instances}) for {n_splits} folds. "
            f"Need at least {n_splits} instances."
        )

    logging.info(
        f"Starting {n_splits}-fold cross-validation on {n_instances} instances"
    )

    # Create KFold splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Storage for per-fold results
    fold_results = []
    fold_metrics = []

    # Iterate over folds
    for fold_num, (train_idx, test_idx) in enumerate(kf.split(instance_paths)):
        logging.info(f"\n{'=' * 60}")
        logging.info(f"Fold {fold_num + 1}/{n_splits}")
        logging.info(f"{'=' * 60}")

        # Get train/test paths
        train_paths = [instance_paths[i] for i in train_idx]
        test_paths = [instance_paths[i] for i in test_idx]

        logging.info(f"Train instances: {len(train_paths)}")
        logging.info(f"Test instances: {len(test_paths)}")

        # Create train/test datasets
        train_data = create_subset_dataset(multi_perf_data, train_paths)
        test_data = create_subset_dataset(multi_perf_data, test_paths)

        # Determine model save location
        if save_models and output_dir:
            model_save_dir = output_dir / f"fold_{fold_num}"
        else:
            # Use temporary directory (models won't be saved)
            import tempfile

            model_save_dir = Path(tempfile.mkdtemp())

        # Train model
        logging.info("Training model...")
        train_pwc(
            train_data,
            save_dir=str(model_save_dir),
            xg_flag=xg_flag,
            feature_csv_path=feature_csv_path,
        )

        # Load trained model
        model_path = model_save_dir / "model.joblib"
        as_model = PwcModel.load(str(model_path))

        # Set feature CSV path if not already set
        if as_model.feature_csv_path is None:
            as_model.feature_csv_path = feature_csv_path

        # Determine output CSV path
        if output_dir:
            output_csv_path = output_dir / f"fold_{fold_num}_results.csv"
        else:
            output_csv_path = None

        # Evaluate model
        logging.info("Evaluating model...")
        result_dataset = as_evaluate(
            as_model,
            test_data,
            write_csv_path=str(output_csv_path) if output_csv_path else None,
        )

        # Compute metrics
        metrics = compute_metrics(result_dataset, test_data)
        fold_metrics.append(metrics)

        # Store fold results
        fold_result = {
            "fold": fold_num + 1,
            "train_size": len(train_paths),
            "test_size": len(test_paths),
            "metrics": metrics,
        }
        fold_results.append(fold_result)

        # Log fold results
        logging.info(f"\nFold {fold_num + 1} Results:")
        logging.info(f"  Solved: {metrics['solved']}/{metrics['total_instances']}")
        logging.info(f"  Solve rate: {metrics['solve_rate']:.2f}%")
        logging.info(f"  Average PAR-2: {metrics['avg_par2']:.2f}")

        # Clean up temporary model if not saving
        if not save_models:
            import shutil

            shutil.rmtree(model_save_dir, ignore_errors=True)

    # Aggregate results across folds
    aggregated = {
        "solve_rate_mean": np.mean([m["solve_rate"] for m in fold_metrics]),
        "solve_rate_std": np.std([m["solve_rate"] for m in fold_metrics]),
        "avg_par2_mean": np.mean([m["avg_par2"] for m in fold_metrics]),
        "avg_par2_std": np.std([m["avg_par2"] for m in fold_metrics]),
        "best_solver_solve_rate_mean": np.mean(
            [m["best_solver_solve_rate"] for m in fold_metrics]
        ),
        "best_solver_solve_rate_std": np.std(
            [m["best_solver_solve_rate"] for m in fold_metrics]
        ),
        "best_solver_avg_par2_mean": np.mean(
            [m["best_solver_avg_par2"] for m in fold_metrics]
        ),
        "best_solver_avg_par2_std": np.std(
            [m["best_solver_avg_par2"] for m in fold_metrics]
        ),
        "virtual_best_solve_rate_mean": np.mean(
            [m["virtual_best_solve_rate"] for m in fold_metrics]
        ),
        "virtual_best_solve_rate_std": np.std(
            [m["virtual_best_solve_rate"] for m in fold_metrics]
        ),
        "virtual_best_avg_par2_mean": np.mean(
            [m["virtual_best_avg_par2"] for m in fold_metrics]
        ),
        "virtual_best_avg_par2_std": np.std(
            [m["virtual_best_avg_par2"] for m in fold_metrics]
        ),
    }

    results = {
        "n_splits": n_splits,
        "n_instances": n_instances,
        "model_type": "XGBoost" if xg_flag else "SVM",
        "random_state": random_state,
        "folds": fold_results,
        "aggregated": aggregated,
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Perform k-fold cross-validation on algorithm selection model"
    )
    parser.add_argument(
        "--perf-csv",
        type=str,
        required=True,
        help="Path to the performance CSV file",
    )
    parser.add_argument(
        "--feature-csv",
        type=str,
        required=True,
        help="Path to the features CSV file",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="Timeout value in seconds (default: 1200.0)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5)",
    )
    parser.add_argument(
        "--xg",
        action="store_true",
        help="Use XGBoost instead of SVM",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results and models (optional)",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save models for each fold (requires --output-dir)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Create output directory if specified
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.save_models:
            logging.info(f"Models will be saved to {output_dir}")
        else:
            logging.info(f"Results will be saved to {output_dir}")

    # Load performance data
    logging.info(f"Loading performance data from {args.perf_csv}")
    multi_perf_data = parse_performance_csv(args.perf_csv, args.timeout)
    logging.info(
        f"Loaded {len(multi_perf_data)} instances with {multi_perf_data.num_solvers()} solvers"
    )

    # Perform cross-validation
    results = cross_validate(
        multi_perf_data,
        args.feature_csv,
        n_splits=args.n_splits,
        xg_flag=args.xg,
        random_state=args.random_state,
        save_models=args.save_models,
        output_dir=output_dir,
        timeout=args.timeout,
    )

    # Print aggregated results
    logging.info("\n" + "=" * 60)
    logging.info("Cross-Validation Results Summary")
    logging.info("=" * 60)
    logging.info(f"Model type: {results['model_type']}")
    logging.info(f"Number of folds: {results['n_splits']}")
    logging.info(f"Total instances: {results['n_instances']}")
    logging.info("")
    logging.info("Algorithm Selection Performance:")
    agg = results["aggregated"]
    logging.info(
        f"  Solve rate: {agg['solve_rate_mean']:.2f}% ± {agg['solve_rate_std']:.2f}%"
    )
    logging.info(
        f"  Average PAR-2: {agg['avg_par2_mean']:.2f} ± {agg['avg_par2_std']:.2f}"
    )
    logging.info("")
    logging.info("Best Single Solver (for comparison):")
    logging.info(
        f"  Solve rate: {agg['best_solver_solve_rate_mean']:.2f}% ± {agg['best_solver_solve_rate_std']:.2f}%"
    )
    logging.info(
        f"  Average PAR-2: {agg['best_solver_avg_par2_mean']:.2f} ± {agg['best_solver_avg_par2_std']:.2f}"
    )
    logging.info("")
    logging.info("Virtual Best Solver (upper bound):")
    logging.info(
        f"  Solve rate: {agg['virtual_best_solve_rate_mean']:.2f}% ± {agg['virtual_best_solve_rate_std']:.2f}%"
    )
    logging.info(
        f"  Average PAR-2: {agg['virtual_best_avg_par2_mean']:.2f} ± {agg['virtual_best_avg_par2_std']:.2f}"
    )
    logging.info("=" * 60)

    # Save results to JSON if output directory specified
    if output_dir:
        results_json_path = output_dir / "cv_results.json"
        with open(results_json_path, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"\nDetailed results saved to {results_json_path}")

        # Also save a summary CSV
        summary_csv_path = output_dir / "cv_summary.csv"
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "fold",
                    "solve_rate",
                    "avg_par2",
                    "best_solver_solve_rate",
                    "best_solver_avg_par2",
                    "virtual_best_solve_rate",
                    "virtual_best_avg_par2",
                ]
            )
            for fold_result in results["folds"]:
                m = fold_result["metrics"]
                writer.writerow(
                    [
                        fold_result["fold"],
                        f"{m['solve_rate']:.2f}",
                        f"{m['avg_par2']:.2f}",
                        f"{m['best_solver_solve_rate']:.2f}",
                        f"{m['best_solver_avg_par2']:.2f}",
                        f"{m['virtual_best_solve_rate']:.2f}",
                        f"{m['virtual_best_avg_par2']:.2f}",
                    ]
                )
            # Add aggregated row
            writer.writerow(
                [
                    "mean",
                    f"{agg['solve_rate_mean']:.2f}",
                    f"{agg['avg_par2_mean']:.2f}",
                    f"{agg['best_solver_solve_rate_mean']:.2f}",
                    f"{agg['best_solver_avg_par2_mean']:.2f}",
                    f"{agg['virtual_best_solve_rate_mean']:.2f}",
                    f"{agg['virtual_best_avg_par2_mean']:.2f}",
                ]
            )
            writer.writerow(
                [
                    "std",
                    f"{agg['solve_rate_std']:.2f}",
                    f"{agg['avg_par2_std']:.2f}",
                    f"{agg['best_solver_solve_rate_std']:.2f}",
                    f"{agg['best_solver_avg_par2_std']:.2f}",
                    f"{agg['virtual_best_solve_rate_std']:.2f}",
                    f"{agg['virtual_best_avg_par2_std']:.2f}",
                ]
            )
        logging.info(f"Summary CSV saved to {summary_csv_path}")


if __name__ == "__main__":
    main()
