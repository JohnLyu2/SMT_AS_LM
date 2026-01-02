"""Text embedding function for SMT descriptions."""

import argparse
import json
import csv
import sys
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# Global model cache
_model_cache = {}


def get_embedding_model(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
) -> SentenceTransformer:
    """
    Get or load an embedding model (cached for efficiency).

    Args:
        model_name: Name of the sentence transformer model to use

    Returns:
        SentenceTransformer model instance
    """
    global _model_cache
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def encode_text(
    text: str | list[str],
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    normalize: bool = False,
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Transform text description(s) into embedding vector(s).

    Args:
        text: Single text string or list of text strings to encode
        model_name: Name of the sentence transformer model to use
        normalize: Whether to normalize embeddings to unit length
        batch_size: Batch size for processing multiple texts
        show_progress: Whether to show progress bar when encoding multiple texts

    Returns:
        numpy array of embeddings:
        - For single text: shape (embedding_dim,)
        - For multiple texts: shape (num_texts, embedding_dim)

    Examples:
        >>> # Single text
        >>> embedding = encode_text("This is a description")
        >>> print(embedding.shape)  # (768,)

        >>> # Multiple texts
        >>> embeddings = encode_text(["Text 1", "Text 2", "Text 3"])
        >>> print(embeddings.shape)  # (3, 768)
    """
    model = get_embedding_model(model_name)

    # Handle single text vs list of texts
    is_single = isinstance(text, str)
    texts = [text] if is_single else text

    # Encode texts
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
        batch_size=batch_size,
        normalize_embeddings=normalize,
    )

    # Return single embedding for single text, array for multiple
    if is_single:
        return embeddings[0]
    return embeddings


def encode_all_desc(
    json_path: str | Path,
    output_csv_path: str | Path | None = None,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    normalize: bool = False,
    batch_size: int = 32,
    show_progress: bool = True,
) -> str:
    """
    Encode all benchmark descriptions from a JSON file and write to CSV.

    Args:
        json_path: Path to JSON file containing benchmark data (e.g., data/raw_jsons/ABV.json)
        output_csv_path: Path to output CSV file. If None, defaults to same directory as JSON
                         with .csv extension (e.g., data/raw_jsons/ABV.csv)
        model_name: Name of the sentence transformer model to use
        normalize: Whether to normalize embeddings to unit length
        batch_size: Batch size for processing multiple texts
        show_progress: Whether to show progress bar when encoding

    Returns:
        Path to the output CSV file

    Examples:
        >>> csv_path = encode_all_desc("data/raw_jsons/ABV.json")
        >>> # Output will be written to data/raw_jsons/ABV.csv
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Determine output path
    if output_csv_path is None:
        output_csv_path = json_path.with_suffix(".csv")
    else:
        output_csv_path = Path(output_csv_path)

    # Load JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    if not benchmarks:
        raise ValueError(f"JSON file is empty or contains no benchmarks: {json_path}")

    # Extract descriptions and smtlib paths
    paths = []
    descriptions = []
    for benchmark in benchmarks:
        smtlib_path = benchmark.get("smtlib_path", "")
        description = benchmark.get("description", "")

        # Use placeholder description if missing or empty
        if not description or not description.strip():
            logic = benchmark.get("logic", "unknown")
            family = benchmark.get("family", "unknown")
            description = f"This a {logic} benchmark from the family {family}"

        paths.append(smtlib_path)
        descriptions.append(description.strip())

    if not descriptions:
        raise ValueError(f"No valid descriptions found in JSON file: {json_path}")

    # Encode all descriptions in batch
    embeddings = encode_text(
        descriptions,
        model_name=model_name,
        normalize=normalize,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    # Get embedding dimension (embeddings is always 2D when encoding multiple texts)
    embedding_dim = embeddings.shape[1]

    # Write to CSV
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        # Create column names: path, emb_0, emb_1, ..., emb_{dim-1}
        fieldnames = ["path"] + [f"emb_{i}" for i in range(embedding_dim)]
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)

        # Write each benchmark's embedding
        for path, embedding in zip(paths, embeddings):
            row = [path] + embedding.tolist()
            writer.writerow(row)

    return str(output_csv_path)


def main():
    """Command-line interface for encode_all_desc."""
    parser = argparse.ArgumentParser(
        description="Encode all benchmark descriptions from a JSON file and write to CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - output will be saved to data/raw_jsons/ABV.csv
  python -m src.desc_encoder data/raw_jsons/ABV.json

  # Specify custom output path
  python -m src.desc_encoder data/raw_jsons/ABV.json -o data/embeddings/ABV.csv

  # Use different model and normalize embeddings
  python -m src.desc_encoder data/raw_jsons/ABV.json --model all-MiniLM-L6-v2 --normalize

  # Adjust batch size and disable progress bar
  python -m src.desc_encoder data/raw_jsons/ABV.json --batch-size 64 --no-progress
        """,
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to JSON file containing benchmark data (e.g., data/raw_jsons/ABV.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file. If not specified, defaults to same directory as JSON with .csv extension",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Name of the sentence transformer model to use (default: sentence-transformers/all-mpnet-base-v2)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize embeddings to unit length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing multiple texts (default: 8)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar when encoding",
    )

    args = parser.parse_args()

    try:
        print(f"Loading JSON file: {args.json_path}")
        csv_path = encode_all_desc(
            json_path=args.json_path,
            output_csv_path=args.output,
            model_name=args.model,
            normalize=args.normalize,
            batch_size=args.batch_size,
            show_progress=not args.no_progress,
        )
        print(f"Success! Embeddings saved to: {csv_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
