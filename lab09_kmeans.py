
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "network_traffic.csv"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

def load_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    return pd.read_csv(DATA)

# Load network traffic dataset
df = load_data()

# Select numerical features
cols = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "failed_logins",
    "connection_count"
]

X = StandardScaler().fit_transform(df[cols])

# 1. Elbow Method
inertias = []

for k in range(1, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 8), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Lab 9 - Elbow Method")
plt.xticks(range(1, 8))
plt.tight_layout()
plt.savefig(OUT / "lab09_elbow.png", dpi=300)
plt.show()
plt.close()

# 2. K-Means Clustering
model = KMeans(n_clusters=2, random_state=42, n_init=10)
df["cluster"] = model.fit_predict(X)

print("\nAverage feature values for each cluster:")
print(df.groupby("cluster")[cols].mean().round(2))

# Save clustered dataset
df.to_csv(OUT / "lab09_clustered_traffic.csv", index=False)

# 3. Visualize clusters using PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=df["cluster"],
    cmap="viridis",
    s=40,
    alpha=0.8
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Lab 9 - K-Means Cluster Visualization")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig(OUT / "lab09_kmeans_clusters.png", dpi=300)
plt.show()
plt.close()

print("\nLab 9 completed successfully!")
print("Elbow graph: reports/lab09_elbow.png")
print("Cluster graph: reports/lab09_kmeans_clusters.png")
print("Clustered data: reports/lab09_clustered_traffic.csv")
