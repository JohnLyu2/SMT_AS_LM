#!/usr/bin/env python3
"""
Find non-incremental benchmarks for a specific logic with ratings in a specified range.

Rating = 1 - (successful_solvers / considered_solvers)
Lower rating means easier (more solvers solved it), higher rating means harder.

Note: Ratings are per evaluation. The most recent evaluation is used by default, or you can specify a specific evaluation ID with --evaluation.

Usage:
    # List available evaluations
    python find_benchmarks_by_rating.py --list-evaluations

    # Find easy benchmarks (rating < 0.5) - uses latest evaluation by default
    python find_benchmarks_by_rating.py --logic QF_BV --max-rating 0.5

    # Find hard benchmarks (rating > 0.7) - uses latest evaluation by default
    python find_benchmarks_by_rating.py --logic QF_BV --min-rating 0.7

    # Find benchmarks in a range (0.5 < rating < 0.8) - uses latest evaluation by default
    python find_benchmarks_by_rating.py --logic QF_BV --min-rating 0.5 --max-rating 0.8

    # Use a specific evaluation ID
    python find_benchmarks_by_rating.py --logic QF_BV --min-rating 0.7 --evaluation 20

    # Export to CSV
    python find_benchmarks_by_rating.py --logic QF_BV --min-rating 0.7 --output results.csv
"""

import sqlite3
import argparse
import csv
import sys
from typing import List, Dict, Optional


def get_latest_evaluation_id(db_path: str) -> Optional[int]:
    """Get the ID of the most recent evaluation."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM Evaluations ORDER BY date DESC, id DESC LIMIT 1")
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


def find_benchmarks(
    db_path: str,
    logic: str,
    min_rating: Optional[float] = None,
    max_rating: Optional[float] = None,
    min_considered_solvers: int = 3,
    evaluation_id: Optional[int] = None,
    output_file: Optional[str] = None,
) -> List[Dict]:
    """
    Find non-incremental benchmarks for a logic with ratings in a specified range.

    Args:
        db_path: Path to SQLite database
        logic: Logic string to filter by
        min_rating: Minimum rating threshold (higher = harder benchmarks)
        max_rating: Maximum rating threshold (lower = easier benchmarks)
        min_considered_solvers: Minimum number of considered solvers (for reliability)
        evaluation_id: Specific evaluation ID to filter by (required)
        output_file: Optional CSV file to write results to

    Returns:
        List of dictionaries with benchmark and rating information
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    cursor = conn.cursor()

    if evaluation_id is None:
        raise ValueError("evaluation_id is required")

    if min_rating is None and max_rating is None:
        raise ValueError(
            "Either --min-rating or --max-rating (or both) must be specified"
        )

    # Build the query - ratings are per evaluation, so we filter by evaluation_id
    query = """
        SELECT DISTINCT
            b.id AS benchmark_id,
            b.name AS benchmark_name,
            b.logic,
            b.category,
            b.size,
            b.queryCount,
            q.id AS query_id,
            q.idx AS query_index,
            r.rating,
            r.consideredSolvers,
            r.successfulSolvers,
            ev.id AS evaluation_id,
            ev.name AS evaluation_name,
            ev.date AS evaluation_date,
            f.id AS family_id,
            f.name AS family_name,
            f.folderName AS family_folderName
        FROM Benchmarks b
        INNER JOIN Queries q ON b.id = q.benchmark
        INNER JOIN Ratings r ON q.id = r.query
        INNER JOIN Evaluations ev ON r.evaluation = ev.id
        LEFT JOIN Families f ON b.family = f.id
        WHERE b.isIncremental = 0
          AND b.logic = ?
          AND r.consideredSolvers >= ?
          AND ev.id = ?
    """

    params = [logic, min_considered_solvers, evaluation_id]

    # Add rating filters
    if min_rating is not None:
        query += " AND r.rating >= ?"
        params.append(min_rating)

    if max_rating is not None:
        query += " AND r.rating <= ?"
        params.append(max_rating)

    query += " ORDER BY r.rating ASC, r.consideredSolvers DESC, b.id, q.idx"

    cursor.execute(query, params)
    results = cursor.fetchall()

    # Convert to list of dictionaries
    benchmarks = []
    for row in results:
        # Construct SMT-LIB path: {logic}/{folderName}/{name}
        folder_name = row["family_folderName"]
        smtlib_path = None
        if folder_name and row["benchmark_name"]:
            smtlib_path = f"{row['logic']}/{folder_name}/{row['benchmark_name']}"

        benchmarks.append(
            {
                "benchmark_id": row["benchmark_id"],
                "benchmark_name": row["benchmark_name"],
                "logic": row["logic"],
                "category": row["category"],
                "size": row["size"],
                "query_count": row["queryCount"],
                "query_id": row["query_id"],
                "query_index": row["query_index"],
                "rating": row["rating"],
                "considered_solvers": row["consideredSolvers"],
                "successful_solvers": row["successfulSolvers"],
                "evaluation_id": row["evaluation_id"],
                "evaluation_name": row["evaluation_name"],
                "evaluation_date": row["evaluation_date"],
                "family_id": row["family_id"],
                "family_name": row["family_name"],
                "family_folderName": folder_name,
                "smtlib_path": smtlib_path,
            }
        )

    conn.close()
    return benchmarks


def print_results(
    benchmarks: List[Dict],
    db_path: str,
    logic: str,
    evaluation_id: int,
    num_solvers: int,
    show_details: bool = False,
    limit: Optional[int] = None,
):
    """Print results to console."""
    if not benchmarks:
        print("No benchmarks found matching the criteria.")
        return

    unique_benchmark_count = len(set(b["benchmark_id"] for b in benchmarks))

    # Get number of unique benchmarks in the evaluation for this logic
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(DISTINCT b.id)
        FROM Benchmarks b
        INNER JOIN Queries q ON b.id = q.benchmark
        INNER JOIN Ratings r ON q.id = r.query
        WHERE b.logic = ?
          AND b.isIncremental = 0
          AND r.evaluation = ?
        """,
        (logic, evaluation_id),
    )
    evaluation_benchmark_count = cursor.fetchone()[0]
    conn.close()

    print(
        f"\nFound {unique_benchmark_count} benchmark(s) among {evaluation_benchmark_count} benchmark(s) in evaluation\n"
    )

    if show_details:
        # Show detailed results
        print(
            f"{'Benchmark ID':<12} {'Query':<8} {'Rating':<8} {'Solvers':<12} {'Logic':<15} {'Family':<25} {'SMT-LIB Path'}"
        )
        print("-" * 150)
        displayed = 0
        for b in benchmarks:
            if limit is not None and displayed >= limit:
                break
            solver_info = f"{b['successful_solvers']}/{b['considered_solvers']}"
            family_name = b.get("family_name") or "N/A"
            smtlib_path = b.get("smtlib_path") or "N/A"
            print(
                f"{b['benchmark_id']:<12} {b['query_index']:<8} {b['rating']:<8.4f} {solver_info:<12} "
                f"{b['logic']:<15} {family_name[:24]:<25} {smtlib_path[:50] if smtlib_path != 'N/A' else 'N/A'}"
            )
            displayed += 1
        if limit is not None and displayed < len(benchmarks):
            print(f"\n... (showing {displayed} of {len(benchmarks)} results)")
    else:
        # Show summary
        unique_benchmarks = {}
        for b in benchmarks:
            bench_id = b["benchmark_id"]
            if bench_id not in unique_benchmarks:
                unique_benchmarks[bench_id] = {
                    "name": b["benchmark_name"],
                    "logic": b["logic"],
                    "category": b["category"],
                    "family_name": b.get("family_name"),
                    "smtlib_path": b.get("smtlib_path"),
                    "queries": [],
                    "min_rating": b["rating"],
                    "max_rating": b["rating"],
                }
            unique_benchmarks[bench_id]["queries"].append(b["query_index"])
            unique_benchmarks[bench_id]["min_rating"] = min(
                unique_benchmarks[bench_id]["min_rating"], b["rating"]
            )
            unique_benchmarks[bench_id]["max_rating"] = max(
                unique_benchmarks[bench_id]["max_rating"], b["rating"]
            )

        print(
            f"{'Benchmark ID':<12} {'Logic':<15} {'Rating':<15} {'Family':<25} {'SMT-LIB Path'}"
        )
        print("-" * 140)
        displayed = 0
        for bench_id, info in sorted(unique_benchmarks.items()):
            if limit is not None and displayed >= limit:
                break
            rating_range = (
                f"{info['min_rating']:.3f}-{info['max_rating']:.3f}"
                if info["min_rating"] != info["max_rating"]
                else f"{info['min_rating']:.3f}"
            )
            family_name = info.get("family_name") or "N/A"
            smtlib_path = info.get("smtlib_path") or "N/A"
            print(
                f"{bench_id:<12} {info['logic']:<15} {rating_range:<15} {family_name[:24]:<25} {smtlib_path[:60] if smtlib_path != 'N/A' else 'N/A'}"
            )
            displayed += 1
        if limit is not None and displayed < len(unique_benchmarks):
            print(f"\n... (showing {displayed} of {len(unique_benchmarks)} benchmarks)")


def write_csv(benchmarks: List[Dict], output_file: str):
    """Write results to CSV file."""
    if not benchmarks:
        print("No benchmarks to write.")
        return

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "benchmark_id",
            "benchmark_name",
            "logic",
            "category",
            "size",
            "rating",
            "evaluation_name",
            "family_name",
            "smtlib_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(benchmarks)

    print(f"\nResults written to {output_file}")


def list_available_logics(db_path: str):
    """List all available logics in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT logic FROM Benchmarks WHERE isIncremental = 0 ORDER BY logic"
    )
    logics = [row[0] for row in cursor.fetchall()]

    print("Available non-incremental logics:")
    for logic in logics:
        cursor.execute(
            "SELECT COUNT(*) FROM Benchmarks WHERE logic = ? AND isIncremental = 0",
            (logic,),
        )
        count = cursor.fetchone()[0]
        print(f"  {logic} ({count} benchmarks)")
    conn.close()


def list_available_evaluations(db_path: str):
    """List all available evaluations in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, name, date, link, 
               (SELECT COUNT(*) FROM Ratings WHERE evaluation = Evaluations.id) as rating_count
        FROM Evaluations 
        ORDER BY date DESC, id DESC
        """
    )
    evaluations = cursor.fetchall()

    print("Available evaluations:")
    print(f"{'ID':<6} {'Date':<12} {'Name':<40} {'Ratings':<10}")
    print("-" * 80)
    for ev_id, name, date, link, rating_count in evaluations:
        name_display = (
            (name[:37] + "...") if name and len(name) > 40 else (name or "N/A")
        )
        date_display = date or "N/A"
        print(f"{ev_id:<6} {date_display:<12} {name_display:<40} {rating_count:<10}")
    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Find non-incremental benchmarks for a logic with ratings in a specified range",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db",
        default="smtlib2025.sqlite",
        help="Path to SQLite database (default: smtlib2025.sqlite)",
    )
    parser.add_argument(
        "--logic", help="Logic string to filter by (e.g., QF_BV, QF_LIA)"
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        help="Minimum rating threshold (higher = harder benchmarks)",
    )
    parser.add_argument(
        "--max-rating",
        type=float,
        help="Maximum rating threshold (lower = easier benchmarks)",
    )
    parser.add_argument(
        "--min-solvers",
        type=int,
        default=3,
        help="Minimum number of considered solvers for reliability (default: 3)",
    )
    parser.add_argument(
        "--evaluation",
        type=int,
        help="Evaluation ID to filter by (ratings are per evaluation). If not specified, uses the most recent evaluation by default.",
    )
    parser.add_argument(
        "--latest-evaluation",
        action="store_true",
        help="Explicitly use the most recent evaluation (this is the default behavior if --evaluation is not specified)",
    )
    parser.add_argument("--output", "-o", help="Optional: Output CSV file path")
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show detailed query-level results instead of benchmark summary",
    )
    parser.add_argument(
        "--list-logics",
        action="store_true",
        help="List all available non-incremental logics and exit",
    )
    parser.add_argument(
        "--list-evaluations",
        action="store_true",
        help="List all available evaluations and exit",
    )

    args = parser.parse_args()

    if args.list_logics:
        list_available_logics(args.db)
        return

    if args.list_evaluations:
        list_available_evaluations(args.db)
        return

    if not args.logic:
        parser.error(
            "--logic is required (unless using --list-logics or --list-evaluations)"
        )

    if args.min_rating is None and args.max_rating is None:
        parser.error("Either --min-rating or --max-rating (or both) must be specified")

    # Determine evaluation ID - default to latest if not specified
    evaluation_id = args.evaluation
    if evaluation_id is None or args.latest_evaluation:
        evaluation_id = get_latest_evaluation_id(args.db)
        if evaluation_id is None:
            parser.error("No evaluations found in database")
        if args.latest_evaluation or args.evaluation is None:
            print(f"Using most recent evaluation (ID: {evaluation_id})")

    # Get evaluation name for display and number of solvers
    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()
    cursor.execute("SELECT name, date FROM Evaluations WHERE id = ?", (evaluation_id,))
    eval_info = cursor.fetchone()
    eval_name = eval_info[0] if eval_info else f"ID {evaluation_id}"
    eval_date = eval_info[1] if eval_info and eval_info[1] else "N/A"

    # Get number of solvers in the evaluation that evaluated this logic
    cursor.execute(
        """
        SELECT COUNT(DISTINCT sv.solver)
        FROM SolverVariants sv
        INNER JOIN Results r ON sv.id = r.solverVariant
        INNER JOIN Queries q ON r.query = q.id
        INNER JOIN Benchmarks b ON q.benchmark = b.id
        WHERE sv.evaluation = ? 
          AND b.logic = ?
          AND b.isIncremental = 0
        """,
        (evaluation_id, args.logic),
    )
    num_solvers = cursor.fetchone()[0]

    # Get total count of non-incremental benchmarks for this logic
    cursor.execute(
        "SELECT COUNT(*) FROM Benchmarks WHERE logic = ? AND isIncremental = 0",
        (args.logic,),
    )
    total_benchmark_count = cursor.fetchone()[0]

    # Get number of unique benchmarks in the evaluation for this logic
    cursor.execute(
        """
        SELECT COUNT(DISTINCT b.id)
        FROM Benchmarks b
        INNER JOIN Queries q ON b.id = q.benchmark
        INNER JOIN Ratings r ON q.id = r.query
        WHERE b.logic = ?
          AND b.isIncremental = 0
          AND r.evaluation = ?
        """,
        (args.logic, evaluation_id),
    )
    evaluation_benchmark_count = cursor.fetchone()[0]
    conn.close()

    # Validate that num_solvers meets the minimum requirement
    if num_solvers < args.min_solvers:
        parser.error(
            f"Number of considered solvers ({num_solvers}) is less than the minimum requirement ({args.min_solvers})"
        )

    print("Searching for non-incremental benchmarks:")
    print(f"  Logic: {args.logic}")
    print(
        f"  Evaluation benchmark size: {evaluation_benchmark_count} (from {total_benchmark_count} SMT-LIB benchmarks)"
    )
    if args.min_rating is not None:
        print(f"  Min rating: {args.min_rating} (harder benchmarks)")
    if args.max_rating is not None:
        print(f"  Max rating: {args.max_rating} (easier benchmarks)")
    print(
        f"  Considered solver num: {num_solvers} >= {args.min_solvers} (the minimum requirement)"
    )
    print(f"  Evaluation: {eval_name} (ID: {evaluation_id}, Date: {eval_date})")
    print()

    try:
        benchmarks = find_benchmarks(
            args.db,
            args.logic,
            args.min_rating,
            args.max_rating,
            args.min_solvers,
            evaluation_id,
            args.output,
        )

        print_results(
            benchmarks, args.db, args.logic, evaluation_id, num_solvers, args.details
        )

        if args.output:
            write_csv(benchmarks, args.output)

    except sqlite3.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
