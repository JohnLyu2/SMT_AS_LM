"""Generate LLM descriptions for benchmarks listed in a JSON file."""

import argparse
import json
import logging
import os
import sys
from tqdm import tqdm
from pathlib import Path

from src.desc_gen_llm import openai_gen_desc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_api_key_from_env_file(env_path: Path) -> bool:
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
                    return True
    except FileNotFoundError:
        return False
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM descriptions for benchmarks listed in a JSON file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_json",
        type=str,
        help="Path to input JSON (e.g., data/raw_jsons/BV.json)",
    )
    parser.add_argument(
        "output_json",
        type=str,
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(REPO_ROOT / "smtlib" / "non-incremental"),
        help="Base directory that contains SMT-LIB benchmarks (default: smtlib/non-incremental)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key. If not provided, uses OPENAI_API_KEY environment variable",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai-gpt-oss-120b",
        help="Model name to use (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://chat-ai.academiccloud.de/v1",
        help="Base URL for the API endpoint.",
    )
    parser.add_argument(
        "--char-limit",
        type=int,
        default=200000,
        help="Maximum number of characters to include from SMT content.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of benchmarks to process",
    )
    return parser.parse_args()


def load_benchmarks(input_json: Path) -> list[dict]:
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {input_json}, got {type(data).__name__}")
    return data


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    if load_api_key_from_env_file(REPO_ROOT / ".env"):
        logger.info("Loaded OPENAI_API_KEY from .env")
    args = parse_args()
    input_json = Path(args.input_json)
    output_json = Path(args.output_json)
    base_dir = Path(args.base_dir)

    benchmarks = load_benchmarks(input_json)
    if args.limit is not None:
        benchmarks = benchmarks[: args.limit]
    results: list[dict] = []
    total = len(benchmarks)

    iterable = tqdm(benchmarks, total=total, desc="Generating", unit="bench")

    for idx, benchmark in enumerate(iterable, start=1):
        smtlib_path = benchmark.get("smtlib_path")
        if not smtlib_path:
            raise ValueError(f"Missing smtlib_path in benchmark: {benchmark}")
        smt_file_path = base_dir / smtlib_path

        result, is_truncated = openai_gen_desc(
            smt_file_path=smt_file_path,
            api_key=args.api_key,
            model=args.model,
            base_url=args.base_url,
            char_limit=args.char_limit,
            prompt_only=False,
        )
        description = result.output_text
        results.append(
            {
                "smtlib_path": smtlib_path,
                "description": description,
                "is_truncated": is_truncated,
            }
        )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
