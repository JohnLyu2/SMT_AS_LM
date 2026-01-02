#!/usr/bin/env python3
"""
Script to encode all benchmark descriptions from JSON files in data/raw_jsons/
and save them to data/features/native_desc/all_mpnet_base_v2/{LOGIC}.csv
"""

import sys
from pathlib import Path

# Add src to path to import desc_encoder
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.desc_encoder import encode_all_desc


def main():
    """Process all JSON files in data/raw_jsons/ and create CSV files."""
    # Define paths
    project_root = Path(__file__).parent.parent
    raw_jsons_dir = project_root / "data" / "raw_jsons"
    output_dir = project_root / "data" / "features" / "native_desc" / "all_mpnet_base_v2"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = sorted(raw_jsons_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {raw_jsons_dir}")
        return 1

    print(f"Found {len(json_files)} JSON file(s) to process")
    print(f"Output directory: {output_dir}\n")

    # Process each JSON file
    for json_file in json_files:
        logic = json_file.stem  # e.g., "ABV" from "ABV.json"
        output_csv = output_dir / f"{logic}.csv"

        print(f"Processing {logic}...")
        print(f"  Input:  {json_file}")
        print(f"  Output: {output_csv}")

        try:
            csv_path = encode_all_desc(
                json_path=str(json_file),
                output_csv_path=str(output_csv),
                model_name="sentence-transformers/all-mpnet-base-v2",
                normalize=False,
                batch_size=8,
                show_progress=True,
            )
            print(f"  Success! Saved to {csv_path}\n")
        except Exception as e:
            print(f"  Error processing {logic}: {e}\n", file=sys.stderr)
            continue

    print("All processing complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

