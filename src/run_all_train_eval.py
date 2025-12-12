import os
import subprocess

# Base folders
base = "./data"
general_data = os.path.join(base, "generalData")

# Feature sources
feature_sources = [ "embeddings_combined"
]


def run_cmd(cmd):
    print("\nRUNNING:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("✔ DONE\n")


# Loop through datasets in generalData
for filename in os.listdir(general_data):
    if not filename.endswith("_train.csv"):
        continue

    logic = filename.replace("_train.csv", "")  # e.g., "bv"
    train_csv = os.path.join(general_data, f"{logic}_train.csv")
    test_csv  = os.path.join(general_data, f"{logic}_test.csv")

    print(f"\n====================================")
    print(f"PROCESSING LOGIC: {logic}")
    print(f"====================================")

    # Now check all feature source folders
    for source in feature_sources:
        source_folder = os.path.join(base, source)

        # Find any matching feature file
        candidates = [
            f for f in os.listdir(source_folder)
            if f.startswith(logic) and f.endswith(".csv")
        ]

        if not candidates:
            print(f"⚠ No feature file found for {logic} in {source}/")
            continue

        feature_csv = os.path.join(source_folder, candidates[0])
        model_dir = f"./models"
        os.makedirs(model_dir, exist_ok=True)

        print(f"\n➡ Training model for:")
        print(f"   Logic:           {logic}")
        print(f"   Feature source:  {source}")
        print(f"   Feature file:    {feature_csv}")
        print(f"   Train file:      {train_csv}")

        # ---------------------
        # TRAIN MODEL
        # ---------------------
        train_cmd = [
            "python3", "-m", "src.pwc",
            "--save-dir", model_dir,
            "--perf-csv", train_csv,
            "--feature-csv", feature_csv
        ]
        run_cmd(train_cmd)

        # ---------------------
        # EVALUATE MODEL
        # ---------------------
        model_path = os.path.join(model_dir, "model.joblib")
        eval_cmd = [
            "python3", "-m", "src.evaluate",
            "--model", model_path,
            "--perf-csv", test_csv
        ]

        print(f"\n➡ Evaluating model for:")
        print(f"   Logic:       {logic}")
        print(f"   Model path:  {model_path}")
        print(f"   Test file:   {test_csv}")

        run_cmd(eval_cmd)

print("\n🎉 ALL TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
