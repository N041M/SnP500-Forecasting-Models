# SP500 Forecasting — Models

**LSTM, Transformer, and hybrid LSTM+Transformer models for S&P 500 forecasting — the code companion to my bachelor's thesis.**

Three generations of architectures, from quick single-file prototypes to a fully instrumented hybrid with post-training diagnostics. The point is the comparison, not a trading edge: the final generation ships a diagnostic suite that distinguishes real directional skill from the market's upward drift — and says so plainly when a model has none.

> Python 3.13 · TensorFlow 2.20 / Keras 3 · yfinance · ta · scikit-learn · pandas · matplotlib + seaborn

---

## Thesis

This repository contains the model code submitted alongside the thesis:

> Grant, R. K. (2026). **Analýza časových řad pomocí rekurentních neuronových sítí pro predikci vývoje indexu S&P 500** *(Time series analysis using recurrent neural networks to predict the development of the S&P 500 index)*. Bachelor's thesis, University of Hradec Králové, Faculty of Informatics and Management.

Full text, methodology, and results: **[theses.cz/id/i8juls](https://theses.cz/id/i8juls/)**

## Model generations

The thesis names the generations **V1.0**, **V2**, and **V3**; the folders match. (Script-internal labels like `V1.1` inside file names are iteration markers independent of the thesis phase numbering — the thesis notes this explicitly.)

### V1.0 — iterative prototyping (`v1.0/`)

Many iterations of essentially the same LSTM predictor (`LSTM/v1.py` → `v1.1.py` → … → `v1.3.py`), plus an early pure-Transformer prototype (`v1transformer.py`) and a side-by-side comparison harness (`v1comparison.py`). The recurring design ideas first appear here: dual price + direction outputs and an asymmetric loss that penalises under-prediction. The scripts compute a rich set of technical indicators, but the thesis-run configuration fed the model a single price feature with a 20-day window; the final V1.0 iteration is `LSTM/v1.1.2.py`.

### V2 — three architectures, one pipeline (`v2/`)

The prototypes consolidated into three clean, comparable implementations sharing a single numpy data pipeline (`SnP500DataPrep.py`, 47 features):

- **Pure LSTM** — `SnP500LSTMV1.py` / `SnP500LSTMV1.1.py`
- **Pure Transformer** — `SnP500TransV1.py` / `SnP500TransV1.1.py`
- **Hybrid LSTM+Transformer** — `SnP500LSTMTransV1.py` / `SnP500LSTMTransV1.1.py`

The `V1.py` / `V1.1.py` suffixes are script-internal revisions (baseline vs. enhanced), not thesis versions. Every change is documented in `PROJECT_CHANGELOG.md`, with design discussion in `WORKFLOW_COMPARISON.md`. Known-broken variants are kept and labelled `_BROKEN` rather than deleted.

### V3 — final hybrid with diagnostics (`v3/`)

A single hybrid LSTM+Transformer (`SnP500LSTMTransV3.py`): bidirectional LSTM layers, three pre-norm Transformer blocks (GELU feed-forward), a custom `FeatureAttention` layer that learns per-feature importance weights, and L2 regularization. The input space grows to 93 features — the technical indicators plus macroeconomic series from FRED (CPI, PPI, the federal funds rate, inflation expectations, Gini coefficient), forward-filled to daily frequency to avoid look-ahead bias (`SnP500DataPrepV3.py`). A post-training analysis suite adds per-split evaluation with confusion matrices, training-dynamics reporting, residual analysis, a trading simulation vs. buy-and-hold, and prediction-collapse detection that flags near-constant outputs and benchmarks direction accuracy against a naive always-up predictor.

## Repository layout

```
v1.0/                 # V1.0 prototype iterations (LSTM/, v1transformer.py, v1comparison.py)
v2/                   # V2: LSTM / Transformer / hybrid + changelogs
v3/                   # V3: final hybrid + diagnostics (config/ holds hyperparameters)
analysis.py           # S&P 500 constituent download + quarterly analysis (yfinance)
preprocessor.py       # index-level data cleaning and preprocessing
testing/              # one-off studies: fourier.py, Prediction.py, Top10.py
*.json                # run configs for the root-level pipelines
```

## Running

All data is downloaded and engineered by the prep scripts — nothing proprietary is required.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# V3 (final generation) — run from inside v3/
cd v3
python SnP500DataPrepV3.py       # download + engineer features → data/processed/
                                 # (macro features need a free FRED API key in
                                 #  config/feature_config.json; runs without one,
                                 #  minus the FRED series)
python SnP500LSTMTransV3.py      # train + full post-training analysis

# V2 — run from inside v2/
cd v2
python SnP500DataPrep.py         # build numpy arrays → data/
python SnP500LSTMV1.py           # or SnP500TransV1.py / SnP500LSTMTransV1.py / *V1.1.py
```
