from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="Loan Classification Demo", page_icon="📊", layout="wide")

# Suppress version-noise warnings in UI logs; proper fix is version pinning in requirements.
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# Support either assignment naming convention: model/ or models/
if (ROOT / "model").exists():
    MODEL_DIR = ROOT / "model"
elif (ROOT / "models").exists():
    MODEL_DIR = ROOT / "models"
else:
    MODEL_DIR = ROOT / "model"

TARGET_COL = "loan_status"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


def get_score_for_auc(model, features):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(features)
    return None


def evaluate_model(model, model_name, data_frame):
    if TARGET_COL not in data_frame.columns:
        raise ValueError(f"Uploaded CSV must contain target column: {TARGET_COL}")

    y_true = data_frame[TARGET_COL].astype(int)
    x_test = data_frame.drop(columns=[TARGET_COL])

    y_pred = model.predict(x_test)
    y_score = get_score_for_auc(model, x_test)

    auc_value = roc_auc_score(y_true, y_score) if y_score is not None else float("nan")

    metrics = {
        "ML Model Name": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc_value,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
    return metrics, y_true, y_pred, report_df


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


st.title("Loan Classification Streamlit App")
st.caption("Evaluate saved classifiers on uploaded test CSV without retraining")

# Keep only the upload-and-test action in orange for clear CTA emphasis.
st.markdown(
    """
    <style>
    button[kind="primary"] {
        background-color: #ff8c00 !important;
        border: 2px solid #cc6f00 !important;
        color: white !important;
        border-radius: 8px !important;
    }
    button[kind="primary"]:hover {
        background-color: #e67e00 !important;
        border: 2px solid #b35f00 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Input")
    selected_model_name = st.selectbox("Select model", list(MODEL_FILES.keys()))
    use_default_clicked = st.button("Test using existing test data", type="primary")

st.subheader("Dataset")
st.write("Choose one test mode:")

default_data_paths = [ROOT / "test_data.csv", DATA_DIR / "test_raw.csv"]
sample_df = None
for sample_path in default_data_paths:
    if sample_path.exists():
        sample_df = pd.read_csv(sample_path)
        break

with st.expander("Sample test CSV format", expanded=True):
    if sample_df is not None:
        st.write("Your uploaded CSV must contain all feature columns plus target column loan_status.")
        st.dataframe(sample_df.head(5), width="stretch")
    else:
        st.info("Sample file not found. Required column: loan_status plus all model feature columns.")

uploaded_file = st.file_uploader("Upload test CSV in the above format", type=["csv"])
use_upload_clicked = st.button("Upload and test new data", type="primary")

test_df = None
if use_default_clicked:
    for fallback_path in default_data_paths:
        if fallback_path.exists():
            test_df = pd.read_csv(fallback_path)
            st.success(f"Loaded existing test data: {fallback_path.name}")
            break
    if test_df is None:
        st.error("No existing test data file found. Expected test_data.csv or data/test_raw.csv")
        st.stop()
elif use_upload_clicked:
    if uploaded_file is None:
        st.warning("Please upload a CSV file first, then click 'Upload and test new data'.")
        st.stop()
    test_df = pd.read_csv(uploaded_file)
    st.success("Uploaded test CSV loaded.")
else:
    st.info("Click one of the two buttons above to start evaluation.")
    st.stop()

st.write(f"Rows: {len(test_df)} | Columns: {len(test_df.columns)}")
st.dataframe(test_df.head(10), width="stretch")

if TARGET_COL not in test_df.columns:
    st.error(f"Target column '{TARGET_COL}' is missing in uploaded file.")
    st.stop()

st.divider()
st.subheader("Single Model Evaluation")
model_path = MODEL_DIR / MODEL_FILES[selected_model_name]
if not model_path.exists():
    st.error(f"Model file not found: {model_path}")
    st.stop()

model = load_model(model_path)
metrics, y_true, y_pred, report_df = evaluate_model(model, selected_model_name, test_df)

auc_display = f"{metrics['AUC']:.4f}" if pd.notna(metrics["AUC"]) else "N/A"
single_model_table = f"""
<table style='width:100%; border-collapse:collapse; margin:0.25rem 0 0.75rem 0;'>
    <tr>
        <td style='border:1px solid #4a4a4a; padding:12px; vertical-align:top; width:25%;'>
            <div style='font-size:18px; color:#dcdcdc;'>Accuracy</div>
            <div style='font-size:56px; line-height:1.1;'> {metrics['Accuracy']:.4f} </div>
        </td>
        <td style='border:1px solid #4a4a4a; padding:12px; vertical-align:top; width:25%;'>
            <div style='font-size:18px; color:#dcdcdc;'>AUC</div>
            <div style='font-size:56px; line-height:1.1;'> {auc_display} </div>
        </td>
        <td style='border:1px solid #4a4a4a; padding:12px; vertical-align:top; width:25%;'>
            <div style='font-size:18px; color:#dcdcdc;'>Precision</div>
            <div style='font-size:56px; line-height:1.1;'> {metrics['Precision']:.4f} </div>
        </td>
        <td style='border:1px solid #4a4a4a; padding:12px; vertical-align:top; width:25%;'>
            <div style='font-size:18px; color:#dcdcdc;'>Recall</div>
            <div style='font-size:56px; line-height:1.1;'> {metrics['Recall']:.4f} </div>
        </td>
    </tr>
    <tr>
        <td style='border:1px solid #4a4a4a; padding:12px; vertical-align:top; width:25%;'>
            <div style='font-size:18px; color:#dcdcdc;'>F1</div>
            <div style='font-size:56px; line-height:1.1;'> {metrics['F1']:.4f} </div>
        </td>
        <td style='border:1px solid #4a4a4a; padding:12px; vertical-align:top; width:25%;'>
            <div style='font-size:18px; color:#dcdcdc;'>MCC</div>
            <div style='font-size:56px; line-height:1.1;'> {metrics['MCC']:.4f} </div>
        </td>
        <td style='border:1px solid #4a4a4a; padding:12px; width:25%;'></td>
        <td style='border:1px solid #4a4a4a; padding:12px; width:25%;'></td>
    </tr>
</table>
"""
st.markdown(single_model_table, unsafe_allow_html=True)

st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(3.0, 2.6))
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
ax.tick_params(labelsize=9)
cm_col1, cm_col2, cm_col3 = st.columns([1, 2, 1])
with cm_col2:
    st.pyplot(fig)

st.subheader("Classification Report")
st.dataframe(report_df, width="stretch")
st.markdown("<div style='border-top:2px solid #4a4a4a; margin:1rem 0 0.25rem 0;'></div>", unsafe_allow_html=True)

st.divider()
st.subheader("All Models Comparison on Current Test Data")
comparison_rows = []
missing_models = []

with st.spinner("Loading model comparison..."):
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            missing_models.append(filename)
            continue

        loaded_model = load_model(path)
        row, _, _, _ = evaluate_model(loaded_model, name, test_df)
        comparison_rows.append(row)

if comparison_rows:
    comparison_df = pd.DataFrame(comparison_rows)
    metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    for col in metric_cols:
        comparison_df[col] = comparison_df[col].round(4)
    comparison_df = comparison_df.sort_values(by="F1", ascending=False).reset_index(drop=True)
    st.dataframe(comparison_df, width="stretch")

    csv_bytes = comparison_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download current comparison as CSV",
        data=csv_bytes,
        file_name="streamlit_model_metrics.csv",
        mime="text/csv",
    )
else:
    st.error("No model files found for comparison.")

if missing_models:
    st.warning("Some model files are missing: " + ", ".join(missing_models))

