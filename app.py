import os
import yaml
import pandas as pd
import streamlit as st
from evaluation import pipeline
from scipy.stats import pearsonr

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(page_title="EduTransAI (Enhanced)", layout="wide")
st.title("EduTransAI (Enhanced) — Interpretable Hybrid MT Evaluation")
st.caption("Neural + baseline metrics • Explainable hybrid scoring • English–Arabic focus")

# -------------------------------
# Sidebar Configuration
# -------------------------------
with st.sidebar:
    st.header("Configuration")
    enable_neural = st.toggle("Enable neural metrics (COMET/BLEURT/BERTScore)", value=False)
    os.environ["ENABLE_NEURAL"] = "true" if enable_neural else "false"

    weights_path = st.text_input("Weights file", "config/weights.yml")
    if st.button("Reload weights"):
        st.cache_data.clear()

# -------------------------------
# Load Weights
# -------------------------------
@st.cache_data
def load_weights(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"Weight file not found: {path}")
        return {}
    except yaml.YAMLError as e:
        st.error(f"Error parsing YAML file: {e}")
        return {}

weights = load_weights(weights_path)

# -------------------------------
# Helper Function
# -------------------------------
def display_scores(scores):
    """Display evaluation scores as JSON."""
    st.json({
        "cosine": scores.cosine,
        "levenshtein": scores.levenshtein,
        "chrf": scores.chrf,
        "bertscore": scores.bertscore,
        "comet": scores.comet,
        "bleurt": scores.bleurt,
        "fluency_heur": scores.fluency_heur,
        "fluency_arabert": scores.fluency_arabert,
        "IHQ": scores.ihq,
    })

# -------------------------------
# Tabs
# -------------------------------
tab_single, tab_batch = st.tabs(["Single pair", "Batch CSV"])

# --- Single Pair Evaluation ---
with tab_single:
    st.subheader("Evaluate a single src–ref–mt triplet")
    col1, col2, col3 = st.columns(3)

    with col1:
        src = st.text_area("Source (EN)", "The minister stressed the importance of regional cooperation.")
    with col2:
        ref = st.text_area("Reference (AR)", "شدد الوزير على أهمية التعاون الإقليمي.")
    with col3:
        mt = st.text_area("MT (AR)", "شدد الوزير على أهمية التعاون الإقليمي.")

    if st.button("Compute scores", type="primary"):
        scores = pipeline.compute_scores(src, mt, ref, weights)
        display_scores(scores)

# --- Batch Evaluation ---
with tab_batch:
    st.subheader("Batch evaluation (CSV)")
    st.write("Upload a CSV with columns: `id,domain,src,ref,mt`. See `data/sample_pairs.csv`.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("data/sample_pairs.csv")

    st.dataframe(df.head())

    if st.button("Run batch"):
        results = []
        for i, row in df.iterrows():
            scores = pipeline.compute_scores(row["src"], row["mt"], row["ref"], weights)
            results.append({
                "id": row.get("id", i),
                "domain": row.get("domain", ""),
                "cosine": scores.cosine,
                "levenshtein": scores.levenshtein,
                "chrf": scores.chrf,
                "bertscore": scores.bertscore,
                "comet": scores.comet,
                "bleurt": scores.bleurt,
                "fluency_heur": scores.fluency_heur,
                "fluency_arabert": scores.fluency_arabert,
                "IHQ": scores.ihq,
            })

        out_df = pd.DataFrame(results)
        st.success("Batch evaluation completed.")
        st.dataframe(out_df)

        # Correlation with human scores if available
        if "human" in df.columns:
            merged = out_df.join(df["human"])
            st.write("If your CSV contains `human` scores, correlations are shown below.")
            for metric in ["cosine", "bertscore", "comet", "bleurt", "IHQ"]:
                if metric in merged and merged[metric].notna().sum() > 3:
                    valid = merged[[metric, "human"]].dropna()
                    r, p = pearsonr(valid[metric], valid["human"])
                    st.write(f"{metric} vs human: r={r:.2f}, p={p:.03f}")

        # Download results
        st.download_button(
            "Download results CSV",
            data=out_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="batch_results.csv",
            mime="text/csv"
        )
