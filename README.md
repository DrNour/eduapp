# EduTransAI (Enhanced)

A Streamlit-based, interpretable MT evaluation toolkit for English–Arabic (and extendable to other pairs), combining:
- **Neural metrics**: COMET, BLEURT, BERTScore
- **Baseline metrics**: chrF, cosine similarity, Levenshtein
- **Fluency**: heuristic + AraBERT-based loss proxy
- **Explainable hybrid score (IHQ)**: weighted, tunable, YAML-configured
- **Visual analytics**: correlation plots, error distributions, domain comparisons

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

> **Note**: Some neural metrics download checkpoints at first run. If running offline or in restricted environments, set `ENABLE_NEURAL=false` in the sidebar or env var to disable them.

## Repo layout
```
app.py
metrics/
  baselines.py
  neural.py
  fluency.py
evaluation/
  pipeline.py
config/
  weights.yml
data/
  sample_pairs.csv
pages/
  1_📊_Batch_Evaluation.py
  2_🧪_Ablations_&_Diagnostics.py
.github/workflows/
  ci.yml
README.md
requirements.txt
```

## IHQ (Interpretable Hybrid Quality)

IHQ = weighted combination of semantic similarity, neural metric(s), and error penalties.
Weights are defined in `config/weights.yml` and can be adjusted from the UI.

## Citation

If you use this toolkit in research or teaching, please cite the corresponding paper or tool documentation.
