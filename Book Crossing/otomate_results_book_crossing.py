import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os

env_python = r"C:\Users\murat\anaconda3\envs\lenskit-env\python.exe"
script_path = "BX_ItemKNN.py"
key_name = os.path.splitext(os.path.basename(script_path))[0]

# Fraction werte: 0.1, 0.2, ..., 1.0
fraction_values = [round(i * 0.1, 1) for i in range(1, 11)]

def run_script(fraction):
    print(f"[{fraction}] wird durchgeführt...")
    result = subprocess.run([env_python, script_path, str(fraction)], capture_output=True, text=True)
    return {
        "fraction": fraction,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }
""" 
# Paralell ausführen wurde entfernt weil der Computer dafür nicht leistungsfähig genug war
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(run_script, frac) for frac in fraction_values]

    for future in as_completed(futures):
        result = future.result()
        print(f"\n--- Fraction {result['fraction']} fertiggestellt ---")
        print("Ausgabe:\n", result["stdout"])
        if result["stderr"]:
            print("Fehler:\n", result["stderr"])
        if result["returncode"] != 0:
            print(f"⚠️ Fraction {result['fraction']} hat einen Fehler ausgegeben!")
"""
# Seriell ausführen
for frac in fraction_values:
    print(f"\n➡️ Starte Skript mit Fraction: {frac}")
    result = subprocess.run([env_python, script_path, str(frac)], capture_output=True, text=True)

    # Die Ergebnisse zeigen
    print("🟢 Ausgabe:\n", result.stdout)
    if result.stderr:
        print("🔴 Fehler:\n", result.stderr)
    if result.returncode != 0:
        print(f"⚠️ Fraction {frac} hat einen Fehlercode zurückgegeben!")

#Nachdem alle Skripte abgeschlossen sind, lade und zeige die Ergebnisse aus der JSON-Datei
print("\n📊 Alle Skripte abgeschlossen. Lade Ergebnisse aus JSON-Datei...")

output_file = "metric_results.json"

try:
    with open(output_file, "r") as f:
        content = json.load(f)

    if key_name not in content:
        print(f"Schlüssel {key_name} wurde nicht gefunden.")
    else:
        print("\nSortierte Ergebnisse:")
        results = content[key_name]

        # Sortiere nach numerischem Wert der Fraktion
        for frac in sorted(results, key=lambda x: float(x)):
            print(f"{frac} → {results[frac]}")

except Exception as e:
    print("Fehler beim Laden der JSON-Datei:", e)