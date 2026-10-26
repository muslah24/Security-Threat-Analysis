
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "network_traffic.csv"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

def load_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    return pd.read_csv(DATA)

# Load dataset
df = load_data()

# Prepare features and target
X = pd.get_dummies(
    df.drop(columns=["threat_label", "attack_type"]),
    dtype=int
)
y = df["threat_label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Train SVM model with feature scaling
model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, gamma="scale")
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

# Print evaluation results
print("Lab 8: SVM Classification")
print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:")
print(classification_report(y_test, pred, zero_division=0))

# Visualization 1: Confusion matrix
fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    pred,
    cmap="Blues",
    values_format="d",
    ax=ax
)
ax.set_title("Lab 8 - SVM Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT / "lab08_svm_confusion_matrix.png", dpi=300)
plt.show()
plt.close()

# Visualization 2: Actual vs predicted class counts
comparison = pd.DataFrame({
    "Actual": y_test.value_counts(),
    "Predicted": pd.Series(pred).value_counts()
}).fillna(0)

comparison.plot(kind="bar", figsize=(8, 5))
plt.title("Lab 8 - Actual vs Predicted Class Counts")
plt.xlabel("Threat Label")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT / "lab08_svm_class_comparison.png", dpi=300)
plt.show()
plt.close()

print("\nVisualizations saved in:", OUT)
