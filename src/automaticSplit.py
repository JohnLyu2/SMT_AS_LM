import os
import subprocess

general_data_folder = "./data/generalData"

for filename in os.listdir(general_data_folder):
    if not filename.endswith(".csv"):
        continue

    # Example: bv.csv → base = "bv"
    base = filename.replace(".csv", "")

    input_path = os.path.join(general_data_folder, filename)
    train_path = os.path.join(general_data_folder, f"{base}_train.csv")
    test_path  = os.path.join(general_data_folder, f"{base}_test.csv")

    cmd = [
        "python3", "-m", "src.split_data",
        "--input", input_path,
        "--train", train_path,
        "--test", test_path
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

print("\n🎉 All datasets have been split successfully!")
