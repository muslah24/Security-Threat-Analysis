
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# ==========================================
# PROJECT PATHS
# ==========================================
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "network_traffic.csv"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)


# ==========================================
# HELPER: WAIT FOR ENTER
# ==========================================
def next_step(message):
    print("\n" + "=" * 60)
    print(message)
    print("=" * 60)
    input("\nPress ENTER to continue...")


# ==========================================
# STEP 1: LOAD DATASET
# ==========================================
print("\nLAB 10: HIERARCHICAL CLUSTERING AND DBSCAN")

if not DATA.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA}")

df = pd.read_csv(DATA)

next_step(
    f"STEP 1: DATASET LOADED\n"
    f"Total records: {len(df)}\n"
    f"Columns: {list(df.columns)}"
)

# ==========================================
# STEP 2: PREPARE FEATURES
# ==========================================
cols = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "failed_logins",
    "connection_count"
]

missing = [col for col in cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
df[cols] = df[cols].fillna(df[cols].median())

scaler = StandardScaler()
X = scaler.fit_transform(df[cols])

next_step(
    "STEP 2: DATA PREPROCESSING COMPLETED\n"
    "Selected five numerical features and standardized them."
)

# ==========================================
# STEP 3: HIERARCHICAL CLUSTERING
# ==========================================
hierarchical = AgglomerativeClustering(n_clusters=2)
df["hierarchical_cluster"] = hierarchical.fit_predict(X)

print("\nHierarchical cluster counts:")
print(df["hierarchical_cluster"].value_counts().sort_index())

next_step(
    "STEP 3: HIERARCHICAL CLUSTERING COMPLETED\n"
    "The records have been grouped into two clusters."
)

# ==========================================
# STEP 4: SHOW DENDROGRAM
# ==========================================
sample_size = min(80, len(X))

if sample_size < 2:
    raise ValueError("At least two records are needed for the dendrogram.")

Z = linkage(X[:sample_size], method="ward")

plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode="lastp", p=20)
plt.title("Lab 10 - Hierarchical Clustering Dendrogram")
plt.xlabel("Cluster / Sample")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(OUT / "lab10_dendrogram.png", dpi=300)

next_step(
    "STEP 4: DENDROGRAM\n"
    "A tree diagram shows how records or groups merge.\n"
    "Close branches indicate merges at smaller distances."
)

plt.show()
plt.close()

# ==========================================
# STEP 5: DBSCAN CLUSTERING
# ==========================================
dbscan = DBSCAN(eps=1.5, min_samples=5)
df["dbscan_cluster"] = dbscan.fit_predict(X)

print("\nDBSCAN cluster counts:")
print(df["dbscan_cluster"].value_counts().sort_index())
print("Label -1 means noise points.")

next_step(
    "STEP 5: DBSCAN COMPLETED\n"
    "DBSCAN groups dense regions and labels noise points as -1."
)

# ==========================================
# STEP 6: VISUALIZE DBSCAN
# ==========================================
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=df["dbscan_cluster"],
    cmap="viridis",
    s=35,
    alpha=0.8
)
plt.title("Lab 10 - DBSCAN Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Cluster (-1 = noise)")
plt.tight_layout()
plt.savefig(OUT / "lab10_dbscan_clusters.png", dpi=300)

next_step(
    "STEP 6: DBSCAN CLUSTER GRAPH\n"
    "Each point represents a record projected into two dimensions.\n"
    "Colors indicate DBSCAN labels; -1 indicates noise."
)

plt.show()
plt.close()

# ==========================================
# STEP 7: COMPARE BOTH METHODS
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

scatter1 = axes[0].scatter(
    X_2d[:, 0], X_2d[:, 1],
    c=df["hierarchical_cluster"],
    cmap="viridis", s=35, alpha=0.8
)
axes[0].set_title("Hierarchical Clustering")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")
fig.colorbar(scatter1, ax=axes[0], label="Cluster")

scatter2 = axes[1].scatter(
    X_2d[:, 0], X_2d[:, 1],
    c=df["dbscan_cluster"],
    cmap="viridis", s=35, alpha=0.8
)
axes[1].set_title("DBSCAN Clustering")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")
fig.colorbar(scatter2, ax=axes[1], label="Cluster (-1 = noise)")

plt.tight_layout()
plt.savefig(OUT / "lab10_clustering_comparison.png", dpi=300)

next_step(
    "STEP 7: CLUSTERING COMPARISON\n"
    "The left graph shows hierarchical clusters.\n"
    "The right graph shows DBSCAN clusters and possible noise."
)

plt.show()
plt.close()

# ==========================================
# STEP 8: SAVE RESULTS
# ==========================================
result_file = OUT / "lab10_cluster_comparison.csv"
df.to_csv(result_file, index=False)

print("\nSTEP 8: RESULTS SAVED")
print("1. reports/lab10_dendrogram.png")
print("2. reports/lab10_dbscan_clusters.png")
print("3. reports/lab10_clustering_comparison.png")
print("4. reports/lab10_cluster_comparison.csv")

print("\nLAB 10 COMPLETED SUCCESSFULLY!")
