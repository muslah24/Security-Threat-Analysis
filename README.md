# Security Threat Analysis — Data Mining Lab Project

## Important note
`network_traffic.csv` and `attack_transactions.csv` are **synthetic educational datasets** created for practicing the lab algorithms. They are not real network captures and must not be presented as real-world measured results.

## Setup in VS Code
1. Open this folder in VS Code.
2. Open Terminal → New Terminal.
3. Create a virtual environment (optional but recommended):
   - Windows: `python -m venv .venv`
   - Activate: `.venv\Scripts\activate`
4. Install libraries: `python -m pip install -r requirements.txt`
5. Run one lab at a time, e.g. `python lab01_python_basics.py`.
6. Generated outputs are stored in `reports/`.

If `python` is not recognized, select the installed Python interpreter in VS Code or install Python 3 from python.org, enable “Add Python to PATH”, then reopen VS Code.

## Dataset columns
- `duration`: connection duration in seconds
- `src_bytes`, `dst_bytes`: bytes sent in each direction
- `failed_logins`: count of failed login attempts
- `connection_count`: connection count in a short observation window
- `protocol`, `service`, `flag`: categorical connection properties
- `threat_label`: Normal or Attack (classification target)
- `attack_type`: illustrative attack category

## Lab mapping
1. Python and CSV
2. Preprocessing
3. EDA and charts
4. Apriori
5. FP-Growth
6. Decision Tree
7. Gaussian Naive Bayes
8. SVM
9. K-Means
10. Hierarchical clustering and DBSCAN
11. Outlier detection and Random Forest
12. End-to-end mini project

## Report outline
For every lab write: Title, Objective, Theory, Tools/Libraries, Algorithm/Procedure, Code, Output/Screenshot, Discussion, Conclusion. Explain your actual output; do not claim a score unless your own run produced it.


## Main Streamlit Application
The project now includes `app.py`, which provides:
- Dashboard and dataset explorer
- Navigation to Labs 1–11, with source viewing and run buttons
- Lab 12 classification comparison: Decision Tree, Gaussian Naive Bayes, and SVM
- Interactive form to predict `Normal` or `Attack` using `threat_label` as the target

### Start the application
In VS Code terminal, from this folder:
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The included dataset is synthetic and intended for educational demonstration only.
