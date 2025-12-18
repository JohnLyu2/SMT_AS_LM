import os
import pandas as pd

features_folder = "./features_filtered"
embeddings_folder = "./embeddings_descriptions"
output_folder = "./features+Embedding"

os.makedirs(output_folder, exist_ok=True)

# Loop through each features file
for feat_file in os.listdir(features_folder):
    if not feat_file.endswith(".csv"):
        continue

    logic_name = feat_file.replace("_features_filtered.csv", "")
    feat_path = os.path.join(features_folder, feat_file)

    # Expected matching embeddings filename
    emb_file = f"{logic_name}_descriptions_embeddings.csv"
    emb_path = os.path.join(embeddings_folder, emb_file)

    if not os.path.exists(emb_path):
        print(f"⚠ No embeddings found for {logic_name}, skipping.")
        continue

    print(f"Processing: {logic_name}")

    # Load both
    df_features = pd.read_csv(feat_path)
    df_embeddings = pd.read_csv(emb_path)

    # Merge on path
    df_merged = df_features.merge(df_embeddings, on="path")

    # Output file
    out_file = os.path.join(output_folder, f"{logic_name}_full_dataset.csv")
    df_merged.to_csv(out_file, index=False)

    print(f"✔ Saved merged dataset: {out_file} (shape={df_merged.shape})")

print("\n🎉 All feature + embedding datasets merged!")
