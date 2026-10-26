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

# Lab 3: EDA and visualization
import matplotlib.pyplot as plt
df = load_data()
df["threat_label"].value_counts().plot(kind="bar", title="Normal vs Attack Traffic")
plt.xlabel("Class"); plt.ylabel("Records"); plt.tight_layout()
plt.savefig(OUT / "lab03_class_counts.png"); plt.close()
df["connection_count"].plot(kind="hist", bins=20, title="Connection Count Distribution")
plt.xlabel("Connection count"); plt.tight_layout()
plt.savefig(OUT / "lab03_connection_histogram.png"); plt.close()
df.boxplot(column="src_bytes", by="threat_label")
plt.suptitle(""); plt.title("Source Bytes by Threat Class"); plt.tight_layout()
plt.savefig(OUT / "lab03_source_bytes_boxplot.png"); plt.close()
corr = df[["duration","src_bytes","dst_bytes","failed_logins","connection_count"]].corr()
plt.imshow(corr, aspect="auto"); plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns); plt.title("Correlation Matrix")
plt.tight_layout(); plt.savefig(OUT / "lab03_correlation.png"); plt.close()
print("Charts saved in reports/. Correlation matrix:\n", corr)
