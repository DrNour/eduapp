import os
import yaml
import pandas as pd
import streamlit as st
from evaluation import pipeline
from scipy.stats import pearsonr

st.set_page_config(page_title="EduTransAI (Enhanced)", layout="wide")
st.title("EduTransAI (Enhanced) — Interpretable Hybrid MT Evaluation")
st.caption("Neural + baseline metrics • Explainable hybrid scoring • English–Arabic focus")

with st.sidebar:
    st.header("Configuration")
    enable_neural = st.toggle("Enable neural metrics (COMET/BLEURT/BERTScore)", value=False)
    os.environ["ENABLE_NEURAL"] = "true" if enable_neural else "false"
    weights_path = st.text_input("Weights file", "config/weights.yml")
    if st.button("Reload weights"):
        st.cache_data.clear()

@st.cache_data
def load_weights(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

weights = load_weights(weights_path)

tab_single, tab_batch = st.tabs(["Single pair", "Batch CSV"])

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
        s = pipeline.compute_scores(src, mt, ref, weights)
        st.json({
            "cosine": s.cosine,
            "levenshtein": s.levenshtein,
            "chrf": s.chrf,
            "bertscore": s.bertscore,
            "comet": s.comet,
            "bleurt": s.bleurt,
            "fluency_heur": s.fluency_heur,
            "fluency_arabert": s.fluency_arabert,
            "IHQ": s.ihq
        })

with tab_batch:
    st.subheader("Batch evaluation (CSV)")
    st.write("Upload a CSV with columns: `id,domain,src,ref,mt`. See `data/sample_pairs.csv`.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        df = pd.read_csv(up)
    else:
        df = pd.read_csv("data/sample_pairs.csv")
    st.dataframe(df.head())

    if st.button("Run batch"):
        rows = []
        for i, r in df.iterrows():
            s = pipeline.compute_scores(r["src"], r["mt"], r["ref"], weights)
            rows.append({
                "id": r.get("id", i),
                "domain": r.get("domain", ""),
                "cosine": s.cosine,
                "levenshtein": s.levenshtein,
                "chrf": s.chrf,
                "bertscore": s.bertscore,
                "comet": s.comet,
                "bleurt": s.bleurt,
                "fluency_heur": s.fluency_heur,
                "fluency_arabert": s.fluency_arabert,
                "IHQ": s.ihq
            })
        out = pd.DataFrame(rows)
        st.success("Done.")
        st.dataframe(out)

        if "human" in df.columns:
            merged = out.join(df["human"])
            st.write("If your CSV contains `human` scores, correlations are shown below.")
            for col in ["cosine","bertscore","comet","bleurt","IHQ"]:
                if col in merged and merged[col].notna().sum() > 3:
                    valid = merged[[col,"human"]].dropna()
                    r, p = pearsonr(valid[col], valid["human"])
                    st.write(f"{col} vs human: r={r:.2f}, p={p:.03f}")
        st.download_button("Download results CSV",
                           data=out.to_csv(index=False).encode("utf-8-sig"),
                           file_name="batch_results.csv", mime="text/csv")
