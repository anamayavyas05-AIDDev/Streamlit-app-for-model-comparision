import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
)

st.set_page_config(page_title="Online Shoppers Purchase Prediction", layout="wide")

st.title("Online Shoppers Purchasing Intention")
st.caption("Predict whether a browsing session ends in a purchase.")

# results.csv is the single source of truth for which models exist
results_df = pd.read_csv("model/results.csv")
model_names = results_df["Model"].tolist()

def model_path(name):
    return f"model/{name.lower().replace(' ', '_')}.pkl"

with st.sidebar:
    st.header("Select Suitable Models")
    choice = st.selectbox("Model", model_names)

DATASET_INFO_PATH = "model/dataset_info.md"
if os.path.exists(DATASET_INFO_PATH):
    with st.expander("About this dataset"):
        with open(DATASET_INFO_PATH) as f:
            st.markdown(f.read())

with st.expander("All models — comparison on the original test set"):
    st.dataframe(results_df, width="stretch")


@st.cache_resource
def load_model(path):
    return joblib.load(path)

st.subheader("1. Upload test data")
uploaded = st.file_uploader("CSV file", type="csv")

if uploaded is None:
    st.info("Upload `test_data.csv` from the repository to see results.")
    st.stop()

# --- 1. is it readable as CSV? ---
try:
    data = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read this file as CSV — {type(e).__name__}: {e}")
    st.stop()

if data.empty:
    st.error("That CSV has no rows.")
    st.stop()

model = load_model(model_path(choice))

# --- 2. does it have the columns the model was trained on? ---
expected = list(model.named_steps["prep"].feature_names_in_)
missing  = [c for c in expected if c not in data.columns]

if missing:
    st.error(f"This CSV is missing {len(missing)} required column(s): "
             f"`{'`, `'.join(missing)}`")
    st.caption("Upload the `test_data.csv` from this repository, "
               "or a CSV with the same schema.")
    st.stop()

st.success(f"Loaded {len(data):,} rows × {data.shape[1]} columns")

TARGET     = "Revenue"
has_target = TARGET in data.columns
X_new      = data[expected]          # exact columns, exact order

# --- 3. does prediction actually work? ---
try:
    pred  = model.predict(X_new)
    proba = model.predict_proba(X_new)[:, 1]
except Exception as e:
    st.error(f"Prediction failed — {type(e).__name__}: {e}")
    st.caption("This usually means a column has an unexpected data type.")
    st.stop()


st.subheader(f"2. Results — {choice}")

if not has_target:
    st.warning(f"No `{TARGET}` column found, so metrics can't be calculated. "
               "Showing predictions only.")
    st.dataframe(pd.DataFrame({"predicted_purchase": pred}).head(50))
    st.stop()

y_true = data[TARGET]

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy",  f"{accuracy_score(y_true, pred):.4f}")
c2.metric("AUC",       f"{roc_auc_score(y_true, proba):.4f}")
c3.metric("Precision", f"{precision_score(y_true, pred, zero_division=0):.4f}")

c4, c5, c6 = st.columns(3)
c4.metric("Recall", f"{recall_score(y_true, pred, zero_division=0):.4f}")
c5.metric("F1",     f"{f1_score(y_true, pred, zero_division=0):.4f}")
c6.metric("MCC",    f"{matthews_corrcoef(y_true, pred):.4f}")

SURFACE, INK, SECOND, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#0d366b"])
LABELS = ["No purchase", "Purchase"]

st.subheader("3. Confusion matrix")

left, right = st.columns([1, 1])

with left:
    cm  = confusion_matrix(y_true, pred)
    cmn = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(cmn, cmap=SEQ, vmin=0, vmax=1)

    for i in range(2):
        for j in range(2):
            dark = cmn[i, j] > 0.42
            ax.text(j, i - 0.09, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=17, fontweight="bold", color="#ffffff" if dark else INK)
            ax.text(j, i + 0.20, f"{cmn[i, j]:.0%}", ha="center", va="center",
                    fontsize=11, color="#cfe0f5" if dark else SECOND)

    ax.set_xticks([0, 1], LABELS, fontsize=10, color=MUTED)
    ax.set_yticks([0, 1], LABELS, fontsize=10, color=MUTED)
    ax.set_xlabel("Predicted", fontsize=10, color=MUTED)
    ax.set_ylabel("Actual", fontsize=10, color=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.text("Classification report")
    st.code(classification_report(y_true, pred, target_names=LABELS, zero_division=0))
