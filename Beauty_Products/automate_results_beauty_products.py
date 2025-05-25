import subprocess
import json
import os

# Python executable path of the conda environment
env_python = r"C:\Users\murat\miniconda3\envs\lenskitenv\python.exe"

# List of script files to run
script_paths = [
    "bpr_beauty_products.py",
    "ease_beauty_products.py",
    "funksvd_beauty_products.py",
    "implicitmf_beauty_products.py",
    "itemknn_lenskit_beauty_products.py",
    "itemknn_recpack_beauty_products.py",
    "kunn_beauty_products.py",
    "nmf_beauty_products.py",
    "popular_beauty_products.py",
    "popularity_beauty_products.py",
    "random_beauty_products.py",
    "slim_beauty_products.py",
    "svd_beauty_products.py",
    "svditemtoitem_beauty_products.py",
    "userknn_beauty_products.py"
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
