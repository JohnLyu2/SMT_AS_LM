import os
import pandas as pd

desc_folder = "./descriptions_full"
feat_folder = "./features_filtered"
output_folder = "./combined"
os.makedirs(output_folder, exist_ok=True)

# Get all descriptionFull CSVs
for desc_file in os.listdir(desc_folder):
    if not desc_file.endswith("descriptions.csv"):
        continue

    logic_name = desc_file.replace("_descriptions.csv", "")
    desc_path = os.path.join(desc_folder, desc_file)

    # Expected matching features file
    feat_file = f"{logic_name}_features_filtered.csv"
    feat_path = os.path.join(feat_folder, feat_file)

    if not os.path.exists(feat_path):
        print(f"⚠ No feature file for {logic_name} → skipping")
        continue

    print(f"Processing {logic_name}: {desc_file} + {feat_file}")

    # Load CSVs
    df_desc = pd.read_csv(desc_path)
    df_feat = pd.read_csv(feat_path)

    # Merge on path
    df = df_desc.merge(df_feat, on="path")

    # Identify which columns are feature columns
    feature_cols = [col for col in df_feat.columns if col != "path"]

    def build_feature_text(row):
        return ", ".join(f"{col}: {row[col]}" for col in feature_cols)

    # Build new description
    df["description"] = df.apply(
        lambda row: f"{row['description']} | {build_feature_text(row)}",
        axis=1
    )

    # Keep only the required columns
    df_final = df[["path", "description"]]

    # Save output
    out_file = os.path.join(output_folder, f"{logic_name}_description_with_features.csv")
    df_final.to_csv(out_file, index=False)

    print(f"✔ Saved {logic_name} → {out_file}")

print("\n🎉 All merges complete!")

