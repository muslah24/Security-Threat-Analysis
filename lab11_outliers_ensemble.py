
# Lab 11: Outlier Detection and Ensemble Methods

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# Set project paths and create reports folder
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

# Load network traffic dataset
df = pd.read_csv(ROOT / "dataset" / "network_traffic.csv")
cols = ["duration", "src_bytes", "dst_bytes", "failed_logins", "connection_count"]

# Prepare and standardize numerical features
X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
X = StandardScaler().fit_transform(X)

# Detect outliers using Isolation Forest (-1 = outlier, 1 = normal)
print("Detecting outliers...")
df["outlier"] = IsolationForest(contamination=0.05, random_state=42).fit_predict(X)
print(df["outlier"].value_counts())

# Visualize detected outliers
plt.scatter(df["src_bytes"], df["dst_bytes"], c=df["outlier"])
plt.title("Network Traffic Outliers")
plt.xlabel("Source Bytes")
plt.ylabel("Destination Bytes")
plt.savefig(OUT / "lab11_outliers.png")
plt.show()

# Find the column containing classification labels
target = next((c for c in ["label", "attack_type", "class", "target"] if c in df.columns), None)
if target is None:
    raise ValueError(f"No target column found. Available columns: {df.columns.tolist()}")

# Prepare target labels and check class counts
y = df[target].fillna("Unknown").astype(str)
if y.nunique() < 2 or y.value_counts().min() < 2:
    raise ValueError("Each class needs at least 2 records.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Create three ensemble classifiers
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "Voting": VotingClassifier([
        ("rf", RandomForestClassifier()),
        ("et", ExtraTreesClassifier())
    ])
}

# Train models and measure accuracy
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results.append({"Model": name, "Accuracy": accuracy_score(y_test, pred)})
    print(name, "accuracy:", round(results[-1]["Accuracy"], 3))

# Save and visualize model accuracy comparison
result = pd.DataFrame(results)
result.to_csv(OUT / "lab11_model_results.csv", index=False)
result.plot(x="Model", y="Accuracy", kind="bar", legend=False, ylim=(0, 1))
plt.title("Ensemble Model Comparison")
plt.tight_layout()
plt.savefig(OUT / "lab11_model_comparison.png")
plt.show()

# Display confusion matrix for the best model
best = max(models, key=lambda n: result.loc[result["Model"] == n, "Accuracy"].iloc[0])
ConfusionMatrixDisplay.from_estimator(models[best], X_test, y_test)
plt.title(f"Confusion Matrix - {best}")
plt.tight_layout()
plt.savefig(OUT / "lab11_confusion_matrix.png")
plt.show()

# Save final dataset with outlier labels
df.to_csv(OUT / "lab11_outlier_results.csv", index=False)
print("Lab 11 completed! Results saved in reports/.")
