import csv
import os

input_folder = "./descriptions"

# Create output folder if you want to separate results
output_folder = "./descriptions_full"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".csv"):
        continue  # skip non-CSV files

    input_file = os.path.join(input_folder, filename)
    output_file = os.path.join(
        output_folder,
        filename.replace(".csv", ".csv")
    )

    print(f"Processing {filename} → {os.path.basename(output_file)}")

    with open(input_file, mode="r", newline="", encoding="utf-8") as infile, \
            open(output_file, mode="w", newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            path = row.get("path", "").strip()
            parts = path.split("/")

            # Extract the second folder (index 1)
            second_folder = parts[1] if len(parts) > 1 else ""

            # If description is empty, fill it
            desc = row.get("description", "")
            if not desc.strip():
                row["description"] = f"this is an instance of {second_folder}"

            writer.writerow(row)

print("\nAll CSV files processed successfully!")
