import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Security Threat Analysis - Lab 12", page_icon="🛡️", layout="wide")
st.title("🛡️ Security Threat Analysis — Lab 12")
st.write("**Data Mining Technique:** Supervised Learning (Classification) | **Models:** Decision Tree, Naive Bayes, SVM")
st.caption("Labs 1–11 remain separate Python files. This interface is only for Lab 12.")

DATA_PATH = "dataset/network_traffic.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_models():
    df = load_data()
    target = "threat_label"
    numeric_cols = ["duration", "src_bytes", "dst_bytes", "failed_logins", "connection_count"]
    categorical_cols = ["protocol", "service", "flag"]
    feature_cols = numeric_cols + categorical_cols
    X = df[feature_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale")
    }
    trained, results, matrices = {}, [], {}
    for name, estimator in models.items():
        preprocessor = ColumnTransformer([
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols)
        ])
        model = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        trained[name] = model
        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, pred), 4),
            "Precision": round(precision_score(y_test, pred, pos_label="Attack", zero_division=0), 4),
            "Recall": round(recall_score(y_test, pred, pos_label="Attack", zero_division=0), 4),
            "F1 Score": round(f1_score(y_test, pred, pos_label="Attack", zero_division=0), 4)
        })
        matrices[name] = confusion_matrix(y_test, pred, labels=["Normal", "Attack"])
    return df, feature_cols, numeric_cols, categorical_cols, trained, pd.DataFrame(results), matrices

try:
    df, feature_cols, numeric_cols, categorical_cols, models, results, matrices = train_models()
except Exception as exc:
    st.error(f"Could not load/train the models: {exc}")
    st.info("Open VS Code in the project root folder and make sure dataset/network_traffic.csv exists.")
    st.stop()

st.subheader("1. Model performance")
st.write("The three models are trained on the included network traffic dataset and evaluated on a held-out test set.")
st.dataframe(results, use_container_width=True, hide_index=True)
best = results.loc[results["F1 Score"].idxmax()]
st.success(f"Best model by Attack F1 Score: {best['Model']} (F1 = {best['F1 Score']:.4f})")

st.divider()
st.subheader("2. Enter network traffic data")
st.write("Enter the connection details below, then click **Analyze Traffic** to get a prediction from all three models.")
with st.form("traffic_input"):
    c1, c2, c3 = st.columns(3)
    with c1:
        duration = st.number_input("Duration", min_value=0, value=10, step=1)
        src_bytes = st.number_input("Source bytes", min_value=0, value=1000, step=100)
        dst_bytes = st.number_input("Destination bytes", min_value=0, value=1000, step=100)
    with c2:
        failed_logins = st.number_input("Failed logins", min_value=0, value=0, step=1)
        connection_count = st.number_input("Connection count", min_value=0, value=5, step=1)
        protocol = st.selectbox("Protocol", sorted(df["protocol"].dropna().astype(str).unique()))
    with c3:
        service = st.selectbox("Service", sorted(df["service"].dropna().astype(str).unique()))
        flag = st.selectbox("Connection flag", sorted(df["flag"].dropna().astype(str).unique()))
        st.caption("Choose values that describe the network connection.")
    submitted = st.form_submit_button("Analyze Traffic", type="primary", use_container_width=True)

if submitted:
    user_data = pd.DataFrame([{
        "duration": duration, "src_bytes": src_bytes, "dst_bytes": dst_bytes,
        "failed_logins": failed_logins, "connection_count": connection_count,
        "protocol": protocol, "service": service, "flag": flag
    }])[feature_cols]
    st.subheader("3. Prediction results")
    cols = st.columns(3)
    for col, (name, model) in zip(cols, models.items()):
        prediction = str(model.predict(user_data)[0])
        with col:
            st.metric(label=name, value=prediction)
            if prediction.lower() == "attack":
                st.warning("Potential threat detected")
            else:
                st.success("Traffic classified as normal")
    st.info("These predictions are model estimates based on the supplied dataset, not a guarantee of real-world security.")

st.divider()
st.subheader("4. Confusion matrices")
st.caption("Rows are actual classes; columns are predicted classes. Class order: Normal, Attack.")
cols = st.columns(3)
for col, (name, matrix) in zip(cols, matrices.items()):
    with col:
        st.markdown(f"**{name}**")
        st.dataframe(pd.DataFrame(matrix, index=["Actual Normal", "Actual Attack"], columns=["Predicted Normal", "Predicted Attack"]), use_container_width=True)

with st.expander("About this Lab 12"):
    st.markdown("""
    - **Application:** Network security threat analysis.
    - **Technique:** Supervised classification.
    - **Dataset:** `dataset/network_traffic.csv`.
    - **Input:** Duration, source/destination bytes, failed logins, connection count, protocol, service and connection flag.
    - **Output:** Normal/Attack predictions from Decision Tree, Naive Bayes and SVM, plus model evaluation metrics.
    """)
