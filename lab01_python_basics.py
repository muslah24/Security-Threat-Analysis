from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "network_traffic.csv"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

def load_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    return pd.read_csv(DATA)

# Lab 1: Python basics and CSV input/output
df = load_data()
print("First 5 rows:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nColumn names:", list(df.columns))
print("\nThreat counts:")
print(df["threat_label"].value_counts())
df.to_csv(OUT / "lab01_loaded_data.csv", index=False)
print("\nSaved reports/lab01_loaded_data.csv")
