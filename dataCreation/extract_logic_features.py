#!/usr/bin/env python3
"""
Extract padded feature vectors for all non-incremental benchmarks of a specific logic.

Outputs a CSV file where:
- First row: "path", feature1, feature2, feature3, ...
- Subsequent rows: smtlib_path, value1, value2, value3, ...

Usage:
    python extract_logic_features.py --db smtlib2025.sqlite --logic QF_NIA --output QF_NIA_features.csv
"""

import sqlite3
import csv
import argparse
import sys
import os
from typing import List, Dict, Tuple, Optional

# Import from local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from get_benchmark_features import get_padded_feature_vector


def get_benchmarks_for_logic(db_path: str, logic: str) -> List[Dict]:
    """
    Get all non-incremental benchmarks for a specific logic with their SMT-LIB paths.

    Args:
        db_path: Path to SQLite database
        logic: Logic string to filter by (e.g., "QF_BV", "QF_NIA")

    Returns:
        List of dictionaries with benchmark information including SMT-LIB paths
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Query to get all non-incremental benchmarks for the logic
    # Join with Families to get the folder name for constructing the path
    query = """
        SELECT
            b.id AS benchmark_id,
            b.name AS benchmark_name,
            b.logic,
            f.folderName AS family_folderName
        FROM Benchmarks b
        LEFT JOIN Families f ON b.family = f.id
        WHERE b.isIncremental = 0
          AND b.logic = ?
        ORDER BY b.id
    """

    cursor.execute(query, (logic,))
    results = cursor.fetchall()
    conn.close()

    benchmarks = []
    for row in results:
        # Construct SMT-LIB path: {logic}/{folderName}/{name}
        folder_name = row["family_folderName"]
        smtlib_path = None
        if folder_name and row["benchmark_name"]:
            smtlib_path = f"{row['logic']}/{folder_name}/{row['benchmark_name']}"

        benchmarks.append({
            "benchmark_id": row["benchmark_id"],
            "benchmark_name": row["benchmark_name"],
            "logic": row["logic"],
            "smtlib_path": smtlib_path,
        })

    return benchmarks


def extract_features_to_csv(
    db_path: str,
    logic: str,
    output_file: str,
    verbose: bool = True
) -> Tuple[int, int]:
    """
    Extract padded feature vectors for all benchmarks in a logic and write to CSV.

    Args:
        db_path: Path to SQLite database
        logic: Logic string to filter by
        output_file: Path to output CSV file
        verbose: Whether to print progress information

    Returns:
        Tuple of (total_benchmarks, successful_extractions)
    """
    if verbose:
        print(f"Extracting features for logic: {logic}")
        print(f"Database: {db_path}")
        print(f"Output: {output_file}\n")

    # Get all benchmarks for the logic
    benchmarks = get_benchmarks_for_logic(db_path, logic)

    if not benchmarks:
        print(f"No non-incremental benchmarks found for logic '{logic}'")
        return 0, 0

    if verbose:
        print(f"Found {len(benchmarks)} non-incremental benchmarks for logic '{logic}'")
        print("Extracting features...\n")

    # Open CSV file for writing
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = None
        successful_count = 0
        failed_count = 0

        for idx, benchmark in enumerate(benchmarks, 1):
            benchmark_name = benchmark['benchmark_name']
            smtlib_path = benchmark['smtlib_path']

            if verbose and idx % 100 == 0:
                print(f"Processing benchmark {idx}/{len(benchmarks)}... ({successful_count} successful, {failed_count} failed)")

            # Skip if no path available
            if not smtlib_path:
                if verbose:
                    print(f"  Warning: Skipping benchmark '{benchmark_name}' - no SMT-LIB path available")
                failed_count += 1
                continue

            # Extract padded feature vector (pass logic to handle duplicate benchmark names)
            feature_values, feature_names = get_padded_feature_vector(db_path, benchmark_name, logic=logic)

            if feature_values is None or feature_names is None:
                if verbose:
                    print(f"  Warning: Could not extract features for '{benchmark_name}'")
                failed_count += 1
                continue

            # Write header on first successful extraction
            if writer is None:
                fieldnames = ['path'] + feature_names
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                if verbose:
                    print(f"CSV header written with {len(feature_names)} features")
                    print(f"Feature vector dimension: {len(feature_names)}\n")

            # Write the row: path, feature_values
            row = [smtlib_path] + feature_values
            writer.writerow(row)
            successful_count += 1

    if verbose:
        print(f"\n{'='*60}")
        print(f"Extraction complete!")
        print(f"{'='*60}")
        print(f"Total benchmarks: {len(benchmarks)}")
        print(f"Successful: {successful_count}")
        print(f"Failed: {failed_count}")
        print(f"Output written to: {output_file}")

    return len(benchmarks), successful_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract padded feature vectors for all non-incremental benchmarks of a specific logic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to SQLite database (e.g., smtlib2025.sqlite)",
    )
    parser.add_argument(
        "--logic",
        required=True,
        help="Logic string to filter by (e.g., QF_BV, QF_NIA)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    try:
        total, successful = extract_features_to_csv(
            args.db,
            args.logic,
            args.output,
            verbose=not args.quiet
        )

        if successful == 0:
            sys.exit(1)

    except sqlite3.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"File I/O error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
