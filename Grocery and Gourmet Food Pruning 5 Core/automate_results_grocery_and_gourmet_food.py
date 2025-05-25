import subprocess
import json
import os

# Python executable path of the conda environment
env_python = r"C:\Users\murat\miniconda3\envs\lenskitenv\python.exe"

# List of script files to run
script_paths = [
    "Bias_Grocery_and_Gourmet_Food.py",
    "biasedmf_grocery_and_gourmet_food.py",
    "bpr_grocery_and_gourmet_food.py",
    "ease_grocery_and_gourmet_food.py",
    "funksvd_grocery_and_gourmet_food.py",
    "implicitmf_grocery_and_gourmet_food.py",
    "itemknn_lenskit_grocery_and_gourmet_food.py",
    "itemknn_recpack_grocery_and_gourmet_food.py",
    "kunn_grocery_and_gourmet_food.py",
    "nmf_grocery_and_gourmet_food.py",
    "popular_grocery_and_gourmet_food.py",
    "popularity_grocery_and_gourmet_food.py",
    "random_grocery_and_gourmet_food.py",
    "slim_grocery_and_gourmet_food.py",
    "SVD_Grocery_and_Gourmet_Food.py",
    "svditemtoitem_grocery_and_gourmet_food.py",
    "userknn_grocery_and_gourmet_food.py"

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
