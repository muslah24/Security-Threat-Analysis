
# Lab 12: End-to-End Data Mining Pipeline

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report

# Set project paths and load dataset
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "dataset" / "network_traffic.csv")
print("Dataset shape:", df.shape)
print(df.head())

# Select features and prepare data
cols = ["duration", "src_bytes", "dst_bytes", "failed_logins", "connection_count"]
X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
target = next((c for c in ["label", "attack_type", "class", "target"] if c in df.columns), None)

# Check that classification labels exist
if target is None:
    raise ValueError(f"No target column found. Columns: {df.columns.tolist()}")
y = df[target].fillna("Unknown").astype(str)
if y.nunique() < 2 or y.value_counts().min() < 2:
    raise ValueError("Need at least two classes, with two records per class.")

# Split dataset and scale numerical features
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the machine-learning model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Evaluate model performance
print("\nAccuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred, zero_division=0))

# Visualize the confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("End-to-End Pipeline: Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT / "lab12_confusion_matrix.png")
plt.show()

# Save predictions and final results
results = pd.DataFrame({"Actual": y_test, "Predicted": pred})
results.to_csv(OUT / "lab12_predictions.csv", index=False)
print("Pipeline completed! Results saved in reports/.")
