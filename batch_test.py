import sys
import os
import subprocess
import re

# === Chemins ===
DATA_DIR = "/home/hemery/code_halt_semi_supervised/data_halt_genkyst/labeled/T2"
MASK_DIR = "/home/hemery/code_halt_semi_supervised/data_halt_genkyst/labeled/mask"
OUTPUT_DIR = "./outputs/"
SCRIPT = "test_semi_supervised.py"

# === Lister les fichiers T2 valides ===
files = [f for f in os.listdir(DATA_DIR) if f.endswith("-T2.nii.gz")]

# === Pattern pour extraire ID et série ===
pattern = re.compile(r"(?P<patient>.+)-(?P<serie>\d{2})-T2\.nii\.gz")

# === Boucle sur les fichiers ===
for f in files:
    match = pattern.match(f)
    if not match:
        print(f" Fichier ignoré (mauvais format) : {f}")
        continue

    patient = match.group("patient")
    serie = int(match.group("serie"))  # Converti '02' → 2

    print(f"  Traitement de : Patient {patient} | Série {serie}")

    # Appel script avec chemin des masques GT passé comme argument
    subprocess.run([
        "python3", SCRIPT,
        "-p", patient,
        "-s", str(serie),
        "-o", OUTPUT_DIR,
        "--gt_dir", MASK_DIR
    ])
