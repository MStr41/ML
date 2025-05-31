import subprocess
import json
import os

# Python executable path of the conda environment
env_python = '/home/frederic/PycharmProjects/Bachelor/venv/bin/python'

# List of script files to run
script_paths = [
    "bias_goodreads_poetry.py",
    #"biasedmf_goodreads_poetry",
    "bpr_goodreads_poetry.py",
    "ease_goodreads_poetry.py",
    "funksvd_goodreads_poetry.py",
    "implicitmf_goodreads_poetry.py",
    "itemknn_lenskit_goodreads_poetry.py",
    "itemknn_recpack_goodreads_poetry.py",
    "kunn_goodreads_poetry.py",
    #"nmf_goodreads_poetry.py",
    "popular_goodreads_poetry.py",
    "popularity_goodreads_poetry.py",
    "random_goodreads_poetry.py",
    "slim_goodreads_poetry.py",
    "svd_goodreads_poetry.py",
    "svditemtoitem_goodreads_poetry.py",
    "userknn_goodreads_poetry.py"
]

# Fractions to try: 0.1, 0.2, ..., 1.0
fraction_values = [round(i * 0.1, 1) for i in range(1, 11)]

# Loop over scripts
for script_path in script_paths:
    key_name = os.path.splitext(os.path.basename(script_path))[0]
    print(f"\n====================\nScript: {script_path}\n====================")

    for frac in fraction_values:
        print(f"\n➡️ Starte {script_path} mit Fraction: {frac}")
        result = subprocess.run([env_python, script_path, str(frac)], capture_output=True, text=True)

        # Output the results
        print("🟢 Ausgabe:\n", result.stdout)
        if result.stderr:
            print("🔴 Fehler:\n", result.stderr)
        if result.returncode != 0:
            print(f"⚠️ Fraction {frac} hat einen Fehlercode zurückgegeben!")

    # Load and print metric results from JSON
    print("\n📊 Lade Ergebnisse aus JSON-Datei...")
    output_file = "metric_results.json"

    try:
        with open(output_file, "r") as f:
            content = json.load(f)

        if key_name not in content:
            print(f"Schlüssel {key_name} wurde nicht gefunden.")
        else:
            print("\nSortierte Ergebnisse:")
            results = content[key_name]
            for frac in sorted(results, key=lambda x: float(x)):
                print(f"{frac} → {results[frac]}")

    except Exception as e:
        print("Fehler beim Laden der JSON-Datei:", e)
