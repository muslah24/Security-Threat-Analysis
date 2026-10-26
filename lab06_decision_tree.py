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

# Lab 6: Decision Tree classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
df = load_data()
X = pd.get_dummies(df.drop(columns=["threat_label","attack_type"]), dtype=int)
y = df["threat_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train); pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
plt.figure(figsize=(16,8)); plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True, max_depth=3)
plt.tight_layout(); plt.savefig(OUT / "lab06_decision_tree.png"); plt.close()
