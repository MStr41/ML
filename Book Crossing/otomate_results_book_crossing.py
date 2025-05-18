# run_external.py
import subprocess
import re
import json
import os

env_python = r"C:\Users\murat\anaconda3\envs\newlenskit\python.exe"
script_path = "BX_ItemKNN.py"

fraction_value = 1.0    

result = subprocess.run([env_python, script_path,str(fraction_value)], capture_output=True, text=True)

print("Ausgabe:\n", result.stdout)
print("Fehler:\n", result.stderr)

#in Zeilen teilen
lines = result.stdout.splitlines()


# metric_results.json Datei lesen
with open("metric_results.json", "r") as f:
    metrics = json.load(f)

if isinstance(metrics, dict):
    try:
        value = list(list(metrics.values())[0].values())[0]
    except (IndexError, AttributeError, TypeError):
        print("Die Json Datei ist nicht in erwünschter Form, die dict ist")
else:
    value = metrics

print(metrics)
print(value)


