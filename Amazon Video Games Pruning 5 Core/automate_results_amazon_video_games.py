import subprocess
import json
import os

# Python executable path of the conda environment
env_python = r"C:\Users\murat\miniconda3\envs\lenskitenv\python.exe"

# List of script files to run
script_paths = [
    "bias_video_games.py",
    "biasedmf_video_games.py",
    "bpr_video_games.py",
    "ease_video_games.py",
    "funksvd_video_games.py",
    "implicitmf_video_games.py",
    "itemknn_lenskit_video_games.py",
    "itemknn_recpack_video_games.py",
    "kunn_video_games.py",
    "nmf_video_games.py",
    "popular_video_games.py",
    "popularity_video_games.py",
    "random_video_games.py",
    "slim_video_games.py",
    "svd_video_games.py",
    "svditemtoitem_video_games.py",
    "userknn_video_games.py"
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
