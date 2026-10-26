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

# Lab 2: Missing values, duplicates, encoding, scaling, statistics
from sklearn.preprocessing import StandardScaler
df = load_data()
print("Missing values before cleaning:\n", df.isna().sum())
df = df.drop_duplicates().copy()
df = df.fillna(df.mode().iloc[0])
categorical = ["protocol", "service", "flag"]
encoded = pd.get_dummies(df, columns=categorical, dtype=int)
numeric = ["duration", "src_bytes", "dst_bytes", "failed_logins", "connection_count"]
encoded[[c + "_scaled" for c in numeric]] = StandardScaler().fit_transform(encoded[numeric])
print("\nDescriptive statistics:\n", df.describe(include="all").T)
encoded.to_csv(OUT / "lab02_preprocessed.csv", index=False)
print("\nSaved reports/lab02_preprocessed.csv")
