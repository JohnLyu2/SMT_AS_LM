import pandas as pd
from pathlib import Path

# Directories
general_data_dir = Path("generalData")
features_dir = Path("features")
output_dir = Path("features_filtered")

output_dir.mkdir(parents=True, exist_ok=True)

# Iterate over all CSV files in generalData
for qflia_file in general_data_dir.glob("*.csv"):
    stem = qflia_file.stem  # e.g. "auflia"
    features_file = features_dir / f"{stem}_features.csv"
    output_file = output_dir / f"{stem}_features_filtered.csv"

    if not features_file.exists():
        print(f"Skipping {stem}: features file not found")
        continue

    # Read generalData file (2-line header, paths in column 0)
    df_qflia = pd.read_csv(qflia_file, skiprows=2, header=None)
    df_qflia = df_qflia.rename(columns={0: "path"})
    df_qflia["path"] = df_qflia["path"].astype(str).str.strip()

    # Read features file
    df_features = pd.read_csv(features_file)
    df_features["path"] = df_features["path"].astype(str).str.strip()

    # Filter features
    df_filtered = df_features[df_features["path"].isin(df_qflia["path"])]

    # Save output
    df_filtered.to_csv(output_file, index=False)

    print(f"Written: {output_file}")
