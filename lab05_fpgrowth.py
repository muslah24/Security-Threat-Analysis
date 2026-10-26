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

# Lab 5: FP-Growth frequent pattern mining
try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import fpgrowth, association_rules
except ImportError:
    raise SystemExit("Install dependencies first: python -m pip install -r requirements.txt")
tx_path = ROOT / "dataset" / "attack_transactions.csv"
transactions = [str(x).split(";") for x in pd.read_csv(tx_path)["items"].dropna()]
te = TransactionEncoder()
basket = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
itemsets = fpgrowth(basket, min_support=0.15, use_colnames=True)
print("FP-Growth itemsets:\n", itemsets)
itemsets.to_csv(OUT / "lab05_fpgrowth_itemsets.csv", index=False)
if len(itemsets) > 0:
    rules = association_rules(itemsets, metric="confidence", min_threshold=0.5)
    rules.to_csv(OUT / "lab05_fpgrowth_rules.csv", index=False)
    print("\nRules:\n", rules[["antecedents","consequents","support","confidence","lift"]])
