import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# -------------------------
# Settings
# -------------------------
folders_to_encode = {
    "./descriptions_full": "./embeddings_descriptions",
    "./combined": "./embeddings_combined"
}

encoder_name = "sentence-transformers/all-mpnet-base-v2"

# -------------------------
# Load model ONCE
# -------------------------
print("Loading model:", encoder_name)
tokenizer = AutoTokenizer.from_pretrained(encoder_name)
encoder = AutoModel.from_pretrained(encoder_name)
encoder.eval()


# Mean pooling function
def mean_pooling(hidden, mask):
    mask = mask.unsqueeze(-1).expand(hidden.size()).float()
    return torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), 1e-9)


# -------------------------
# Process a single CSV file
# -------------------------
def encode_csv(input_path, output_path):
    print(f"Encoding: {input_path}")

    df = pd.read_csv(input_path)

    # Normalize descriptions
    df["description"] = (
        df["description"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    rows = []

    with torch.no_grad():
        for i, row in df.iterrows():
            text = str(row["description"])
            path = row["path"]

            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256
            )

            # Encode
            outputs = encoder(**inputs)
            hidden = outputs.last_hidden_state
            emb = mean_pooling(hidden, inputs["attention_mask"])  # shape (1,768)

            vec = emb.squeeze(0).cpu().numpy()
            rows.append([path] + vec.tolist())

    # Build column names
    columns = ["path"] + [f"emb_{i}" for i in range(len(vec))]

    # Save output CSV
    out_df = pd.DataFrame(rows, columns=columns)
    out_df.to_csv(output_path, index=False)

    print(f"✔ Saved: {output_path} | shape={out_df.shape}")


# -------------------------
# Loop through folders
# -------------------------
for in_folder, out_folder in folders_to_encode.items():
    os.makedirs(out_folder, exist_ok=True)

    print("\n===============================================")
    print(f"Processing folder: {in_folder}")
    print("===============================================\n")

    for filename in os.listdir(in_folder):
        if not filename.endswith(".csv"):
            continue

        input_path = os.path.join(in_folder, filename)

        out_name = filename.replace(".csv", "_embeddings.csv")
        output_path = os.path.join(out_folder, out_name)

        encode_csv(input_path, output_path)

print("\n🎉 ALL EMBEDDING FILES GENERATED SUCCESSFULLY!")
