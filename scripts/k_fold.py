#!/usr/bin/env python3
"""
Script to split a performance CSV file into k folds.

Each fold is saved as a separate CSV file in the output folder.
"""

import csv
import argparse
from pathlib import Path
from sklearn.model_selection import KFold


def split_into_folds(input_path, output_dir, n_splits=5, random_state=42):
    """
    Split a CSV file into k folds and save each fold as a separate file.

    Args:
        input_path: Path to input CSV file
        output_dir: Directory to save fold CSV files
        n_splits: Number of folds (default: 5)
        random_state: Random seed for reproducibility (default: 42)
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the CSV file
    with input_path.open(mode="r", newline="") as csv_file:
        csv_reader = csv.reader(csv_file)

        # Read header rows (first two rows)
        header1 = next(csv_reader)
        header2 = next(csv_reader)

        # Read all data rows
        data_rows = list(csv_reader)

    n_instances = len(data_rows)
    if n_instances < n_splits:
        raise ValueError(
            f"Not enough instances ({n_instances}) for {n_splits} folds. "
            f"Need at least {n_splits} instances."
        )

    # Create KFold splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Split into folds and save each fold
    for fold_num, (train_idx, test_idx) in enumerate(kf.split(data_rows)):
        # For k-fold, we typically want to save each fold as a test set
        # But we can also save the complement (train set) if needed
        # Here we'll save the test fold (the held-out fold)
        fold_rows = [data_rows[i] for i in test_idx]

        # Save fold as CSV
        fold_path = output_dir / f"{fold_num}.csv"
        with fold_path.open(mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(header1)
            csv_writer.writerow(header2)
            csv_writer.writerows(fold_rows)

        print(
            f"Fold {fold_num + 1}/{n_splits}: {len(fold_rows)} instances -> {fold_path}"
        )

    print("\nSplit complete!")
    print(f"  Total instances: {n_instances}")
    print(f"  Number of folds: {n_splits}")
    print(f"  Instances per fold: ~{n_instances // n_splits}")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Split a performance CSV file into k folds"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to save fold CSV files",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=4,
        help="Number of folds (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    split_into_folds(
        args.input,
        args.output_dir,
        n_splits=args.n_splits,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
