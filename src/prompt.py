"""Functions for creating description prompts from SMT files."""

import json
from pathlib import Path

BASIC_INFO_FIELDS = [
    "family",
    "smtlib_path",
    "asserts_count",
    "declare_fun_count",
    "declare_const_count",
    "declare_sort_count",
    "define_fun_count",
    "define_fun_rec_count",
    "constant_fun_count",
    "define_sort_count",
    "declare_datatype_count",
    "max_term_depth",
]


def get_basic_info_from_json(
    smtlib_path: str, json_path: str | Path = "data/raw_jsons/BV.json"
) -> tuple[dict, dict]:
    """
    Get basic information for an SMT benchmark from JSON.

    Args:
        smtlib_path: SMT-LIB path to match (e.g., "BV/2017-Preiner-scholl-smt08/RND/RND_3_15.smt2")
        json_path: Path to the JSON file containing benchmark data (default: "data/raw_jsons/BV.json")

    Returns:
        Tuple of (basic_info, symbol_counts):
        - basic_info: Dictionary of basic info fields from BASIC_INFO_FIELDS
        - symbol_counts: Dictionary of symbol counts (only non-zero symbols)

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        ValueError: If the benchmark is not found in the JSON file
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    # Find matching benchmark by smtlib_path
    matched_benchmark = None
    for benchmark in benchmarks:
        if "smtlib_path" in benchmark and benchmark["smtlib_path"] == smtlib_path:
            matched_benchmark = benchmark
            break

    if matched_benchmark is None:
        raise ValueError(f"Benchmark not found in JSON for smtlib_path: {smtlib_path}")

    # Extract basic info fields
    basic_info = {}
    for field in BASIC_INFO_FIELDS:
        basic_info[field] = matched_benchmark[field]

    # Extract symbol counts
    symbol_counts = {}
    if "symbol_counts" in matched_benchmark and matched_benchmark["symbol_counts"]:
        symbol_counts = {
            symbol: count
            for symbol, count in matched_benchmark["symbol_counts"].items()
            if count > 0
        }

    return basic_info, symbol_counts


def format_basic_info_text(basic_info: dict, symbol_counts: dict) -> str:
    """
    Format basic info and symbol counts into a text string for the prompt.

    Args:
        basic_info: Dictionary of basic info field defined in BASIC_INFO_FIELDS
        symbol_counts: Dictionary of symbol counts

    Returns:
        Formatted text string with basic information
    """
    text = "\n\nBasic Information about the instance:\n"
    if basic_info:
        if "family" in basic_info:
            text += f"family: {basic_info['family']}\n"

        if "smtlib_path" in basic_info:
            text += f"smtlib_path: {basic_info['smtlib_path']}\n"

        # Count fields (skip zero values)
        count_fields = [
            "asserts_count",
            "declare_fun_count",
            "declare_const_count",
            "declare_sort_count",
            "define_fun_count",
            "define_fun_rec_count",
            "constant_fun_count",
            "define_sort_count",
            "declare_datatype_count",
        ]
        for field in count_fields:
            if field in basic_info and basic_info[field] > 0:
                display_name = field.replace("_count", " count").replace("_", "-")
                if display_name == "asserts count":
                    display_name = "assert count"
                text += f"{display_name}: {basic_info[field]}\n"

        # Last line: max_term_depth
        if "max_term_depth" in basic_info:
            text += f"max_term_depth: {basic_info['max_term_depth']}\n"

    if symbol_counts:
        # Show all symbols sorted by count
        sorted_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)
        symbols_str = ", ".join(f"{k}: {v}" for k, v in sorted_symbols)
        text += f"\nSymbol counts: {symbols_str}\n"
    return text


def create_prompt_from_smt_file(
    smt_file_path: str | Path, char_limit: int = 20000
) -> str:
    """
    Read an SMT file and create a prompt for generating a description.

    Args:
        smt_file_path: Path to the SMT file (.smt2)
        char_limit: Maximum number of characters to include from SMT content (default: 20000).
                   Content exceeding this limit will be truncated.

    Returns:
        Prompt string for the LLM

    Raises:
        FileNotFoundError: If the SMT file doesn't exist
        ValueError: If the SMT file is empty
    """
    smt_path = Path(smt_file_path)
    if not smt_path.exists():
        raise FileNotFoundError(f"SMT file not found: {smt_path}")

    # Extract smtlib_path by keeping only the part after "non-incremental/"
    smt_path_str = str(smt_path)
    if "non-incremental/" not in smt_path_str:
        raise ValueError(f"Path must contain 'non-incremental/': {smt_path}")
    smtlib_path = smt_path_str.split("non-incremental/", 1)[1]

    # Get basic info from JSON
    basic_info_dict, symbol_counts = get_basic_info_from_json(smtlib_path)

    # Read SMT file content
    with open(smt_path, "r", encoding="utf-8") as f:
        smt_content = f.read().strip()

    if not smt_content:
        raise ValueError(f"SMT file is empty: {smt_path}")

    # Truncate if exceeds char_limit
    if len(smt_content) > char_limit:
        smt_content = smt_content[:char_limit]

    # Build basic info string
    basic_info_str = format_basic_info_text(basic_info_dict, symbol_counts)

    # Create prompt
    prompt = f"""Analyze the following SMT-LIB instance and provide a concise description of what it represents. 
Focus on:
- The logic/theory used
- The main problem structure
- Key constraints or properties being checked
- Any notable characteristics

SMT-LIB instance:
```
{smt_content}
```
{basic_info_str}
"""
    return prompt
