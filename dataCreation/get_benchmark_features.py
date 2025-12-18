#!/usr/bin/env python3
"""
Simple feature extraction for individual SMT-LIB benchmarks.

Given a benchmark filename, returns symbol counts and command features.
"""

import sqlite3
from typing import Dict, Any, Optional, Tuple, List
import sys
import os

# Import logic_features module from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logic_features import get_relevant_features

"""
Get features for a single non-incremental benchmark by name.

Args:
    db_path: Path to SQLite database
    benchmark_name: Benchmark filename (e.g., "calc2_sec2_bmc25.smt2")

Returns:
    Dictionary with:
    {
        'benchmark_id': int,
        'benchmark_name': str,
        'logic': str,
        'query_id': int,
        'status': str,
        'symbol_counts': {symbol_name: count, ...},
        'commands': {
            'assert': count,
            'declare-fun': count,
            'declare-const': count,
            'declare-sort': count,
            'define-fun': count,
            'define-fun-rec': count,
            'define-funs-rec': count,
            'define-sort': count,
            'declare-datatype': count,
            'declare-datatypes': count
        },
        'metadata': {
            'maxTermDepth': int,
            'normalizedSize': int,
        }
    }

    Returns None if benchmark not found or is incremental.

Example:
    >>> features = get_benchmark_features('smtlib2025.sqlite', 'calc2_sec2_bmc25.smt2')
    >>> print(features['logic'])
    'QF_NIA'
    >>> print(features['symbol_counts']['and'])
    42
    >>> print(features['commands']['assert'])
    10
"""


def get_benchmark_features(
        db_path: str,
        benchmark_name: str,
        logic: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get benchmark and query info (1:1 mapping for non-incremental)
    query = """
            SELECT b.id   as benchmark_id, \
                   b.name as benchmark_name, \
                   b.logic, \
                   q.id   as query_id, \
                   q.status, \
                   q.inferredStatus, \
                   q.assertsCount, \
                   q.maxTermDepth, \
                   q.normalizedSize, \
                   q.declareFunCount, \
                   q.declareConstCount, \
                   q.declareSortCount, \
                   q.defineFunCount, \
                   q.defineFunRecCount, \
                   q.constantFunCount, \
                   q.defineSortCount, \
                   q.declareDatatypeCount
            FROM Benchmarks b
                     JOIN Queries q ON q.benchmark = b.id
            WHERE b.name = ? \
              AND b.isIncremental = 0 \
            """

    params = [benchmark_name]

    # Add logic filter if provided (important when benchmark names are not unique across logics)
    if logic is not None:
        query += " AND b.logic = ?"
        params.append(logic)

    query += " LIMIT 1"

    cursor.execute(query, params)
    result = cursor.fetchone()

    if not result:
        conn.close()
        return None

    # Unpack results
    (benchmark_id, benchmark_name, logic, query_id, status, inferred_status,
     asserts_count, max_term_depth, normalized_size,
     declare_fun_count, declare_const_count, declare_sort_count,
     define_fun_count, define_fun_rec_count, constant_fun_count,
     define_sort_count, declare_datatype_count) = result

    # Get symbol counts
    symbol_query = """
                   SELECT s.name, sc.count
                   FROM SymbolCounts sc
                            JOIN Symbols s ON sc.symbol = s.id
                   WHERE sc.query = ? \
                   """

    cursor.execute(symbol_query, (query_id,))
    symbol_counts = {name: count for name, count in cursor.fetchall()}

    conn.close()

    # Build command features dictionary
    # Map from Queries table columns to command names
    commands = {
        'assert': asserts_count or 0,
        'declare-fun': declare_fun_count or 0,
        'declare-const': declare_const_count or 0,
        'declare-sort': declare_sort_count or 0,
        'define-fun': define_fun_count or 0,
        'define-fun-rec': define_fun_rec_count or 0,  # Includes define-funs-rec
        'constant-fun': constant_fun_count or 0,  # 0-ary define-fun (constants)
        'define-sort': define_sort_count or 0,
        'declare-datatype': declare_datatype_count or 0,  # Includes declare-datatypes
    }

    # Build metadata dictionary (only non-redundant fields)
    metadata = {
        'maxTermDepth': max_term_depth or 0,
        'normalizedSize': normalized_size or 0,
    }

    # Return complete feature dictionary
    return {
        'benchmark_id': benchmark_id,
        'benchmark_name': benchmark_name,
        'logic': logic,
        'query_id': query_id,
        'status': status or 'unknown',
        'inferredStatus': inferred_status or 'unknown',
        'symbol_counts': symbol_counts,
        'commands': commands,
        'metadata': metadata,
    }


def get_padded_feature_vector(
        db_path: str,
        benchmark_name: str,
        logic: Optional[str] = None
) -> Tuple[Optional[List[int]], Optional[List[str]]]:
    """
    Get a fixed-size, zero-padded feature vector for a benchmark.

    Returns a feature vector where:
    - All features relevant to the benchmark's logic are included
    - Features present in the benchmark have their actual counts
    - Features not present in the benchmark are padded with 0
    - Feature ordering is consistent across all benchmarks of the same logic

    Args:
        db_path: Path to SQLite database
        benchmark_name: Benchmark filename (e.g., "calc2_sec2_bmc25.smt2")
        logic: Optional logic filter (recommended when benchmark names are not unique)

    Returns:
        Tuple of (feature_values, feature_names):
        - feature_values: List of counts/values in fixed order
        - feature_names: List of feature names in same order

        Returns (None, None) if benchmark not found or is incremental.

    Example:
        >>> values, names = get_padded_feature_vector('smtlib2025.sqlite', 'calc2_sec2_bmc25.smt2', 'QF_NIA')
        >>> print(len(values), len(names))  # Same length
        >>> print(values[:5])  # First 5 feature values
        [10, 5, 0, 2, 15]
        >>> print(names[:5])   # First 5 feature names
        ['assert', 'declare-fun', 'declare-const', 'define-fun', 'and']
    """
    # Get raw benchmark features (with logic filter to handle duplicate names)
    features = get_benchmark_features(db_path, benchmark_name, logic=logic)

    if features is None:
        return None, None

    # Get the logic
    logic = features['logic']

    # Get all relevant features for this logic (commands + theory operators + metadata)
    relevant_features = get_relevant_features(logic, include_commands=True, include_metadata=True)

    # Merge all actual features into a single dictionary
    actual_features = {}

    # Add command counts
    actual_features.update(features['commands'])

    # Add symbol counts
    actual_features.update(features['symbol_counts'])

    # Add metadata
    actual_features.update(features['metadata'])

    # Build the padded feature vector
    feature_values = []
    feature_names = []

    for feature_name in relevant_features:
        feature_names.append(feature_name)
        # Use actual count if present, otherwise 0
        feature_values.append(actual_features.get(feature_name, 0))

    return feature_values, feature_names


def main():
    """Example usage."""
    import sys
    import json

    if len(sys.argv) < 3:
        print("Usage: python get_benchmark_features.py <db_path> <benchmark_name>")
        print("\nExample:")
        print("  python get_benchmark_features.py smtlib2025.sqlite calc2_sec2_bmc25.smt2")
        return 1

    db_path = sys.argv[1]
    benchmark_name = sys.argv[2]

    features = get_benchmark_features(db_path, benchmark_name)

    if features is None:
        print(f"ERROR: Benchmark '{benchmark_name}' not found or is incremental")
        return 1

    # Print features in a nice format
    print(f"Benchmark: {features['benchmark_name']}")
    print(f"Logic: {features['logic']}")
    print(f"Status: {features['status']}")
    print(f"\n=== Commands ===")
    for cmd, count in sorted(features['commands'].items()):
        if count > 0:
            print(f"  {cmd:20s}: {count:6,}")

    print(f"\n=== Top Symbol Counts ===")
    top_symbols = sorted(features['symbol_counts'].items(),
                         key=lambda x: x[1], reverse=True)[:10]
    for symbol, count in top_symbols:
        print(f"  {symbol:20s}: {count:6,}")

    print(f"\n=== Metadata ===")
    for key, value in sorted(features['metadata'].items()):
        print(f"  {key:20s}: {value:6,}")

    print(f"\n=== Full JSON Output ===")
    print(json.dumps(features, indent=2, default=str))

    # Demonstrate padded feature vector
    print(f"\n{'=' * 60}")
    print("=== Padded Feature Vector ===")
    print(f"{'=' * 60}")

    feature_values, feature_names = get_padded_feature_vector(db_path, benchmark_name)

    if feature_values is not None:
        print(f"\nTotal features for logic '{features['logic']}': {len(feature_values)}")
        print(f"Non-zero features: {sum(1 for v in feature_values if v > 0)}")
        print(f"Zero-padded features: {sum(1 for v in feature_values if v == 0)}")

        print(f"\n=== First 10 Features ===")
        for i in range(min(10, len(feature_names))):
            print(f"  [{i:3d}] {feature_names[i]:20s}: {feature_values[i]:6,}")

        print(f"\n=== Non-zero Features ===")
        for i, (name, value) in enumerate(zip(feature_names, feature_values)):
            if value > 0:
                print(f"  [{i:3d}] {name:20s}: {value:6,}")

    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main())