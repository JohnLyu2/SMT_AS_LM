import subprocess


logics = [
"QF_LRA"
]

for logic in logics:
    out_file = f"./generalData/{logic.lower()}.csv"
    cmd = [
        "python3", "createRuntimeCsv.py",
        "--db", "./SMT-LIB-Catalog-2025/smtlib2025.sqlite",
        "--logic", logic,
        "--output-format", "solver",
        "--output", out_file,
        "--min-rating", "0"
    ]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(e)


    out_file = f"./descriptions/{logic.lower()}_descriptions.csv"
    cmd = [
        "python3", "createRuntimeCsv.py",
        "--db", "./SMT-LIB-Catalog-2025/smtlib2025.sqlite",
        "--logic", logic,
        "--output-format", "description",
        "--output", out_file,
        "--min-rating", "0"
    ]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(e)



    out_file = f"./features/{logic.lower()}_features.csv"
    cmd = [
        "python3", "extract_logic_features.py",
        "--db", "./SMT-LIB-Catalog-2025/smtlib2025.sqlite",
        "--logic", logic,
        "--output", out_file]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(e)


subprocess.run(
    ["python3", "adjustDescriptions.py"],
    check=True
)

subprocess.run(
    ["python3", "filterFeaturesWithGeneralData.py"],
    check=True
)

subprocess.run(
    ["python3", "createFeaturesAsDescription.py"],
    check=True
)


subprocess.run(
    ["python3", "encoder.py"],
    check=True
)

subprocess.run(
    ["python3", "combineFeaturesEmbeddings.py"],
    check=True
)

print("Done!")
