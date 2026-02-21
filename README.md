# 🪙 Bitcoin Direction Prediction with Gradient Boosting

A end-to-end machine learning pipeline for predicting the monthly direction of **Bitcoin (BTC-USD)** using Gradient Boosting and rigorous financial backtesting methodology.

---

## 📋 Overview

This project builds a binary classifier that predicts whether Bitcoin will close higher or lower the following month, then translates those predictions into a long/flat trading strategy and evaluates it against a buy-and-hold benchmark. The focus is equally on **predictive modelling** and **methodological correctness** — avoiding the data leakage pitfalls that are common in financial ML projects.

Bitcoin was chosen over traditional equity indices (e.g. S&P 500) because crypto markets are less informationally efficient, momentum and volatility features carry stronger predictive signal, and the higher volatility creates more differentiated market regimes for the model to learn from.

---

## 🗂️ Project Structure

```
btc-gbm-prediction/
├── data/
│   ├── btc_features.csv            ← Engineered feature dataset
│   ├── cv_results.csv              ← Walk-forward CV results
│   ├── feature_importance.csv      ← Fold-averaged importances (leak-free)
│   ├── model_comparison.csv        ← Baseline comparison table
│   ├── best_params.csv             ← Best hyperparameters from nested CV
│   ├── final_metrics.csv           ← Head-to-head performance metrics
│   └── backtest_results.csv        ← Full monthly backtest time series
├── models/
│   ├── gbm_btc.joblib              ← sklearn GBM (trained on first 80%)
│   └── lgbm_optimized.joblib       ← LightGBM optimised (trained on first 80%)
├── notebooks/
│   ├── 01_data_eda.ipynb           ← Data download, feature engineering & EDA
│   ├── 02_gbm_model.ipynb          ← GBM training, walk-forward CV & baselines
│   ├── 03_hyperparameter_tuning.ipynb  ← LightGBM + Nested CV tuning
│   └── 04_backtesting_strategy.ipynb   ← Signal generation & financial backtest
└── README.md
```

---

## 📓 Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_data_eda.ipynb` | Downloads BTC-USD from Yahoo Finance (2015–2024), engineers 14 technical features, and performs exploratory analysis including seasonality and correlation structure |
| 2 | `02_gbm_model.ipynb` | Trains a `GradientBoostingClassifier` with walk-forward CV, compares against Dummy, Logistic Regression and Random Forest baselines, and computes leak-free feature importances |
| 3 | `03_hyperparameter_tuning.ipynb` | Migrates to LightGBM and performs **Nested Cross-Validation** (inner tuning loop + outer evaluation loop) for honest, bias-corrected performance estimation |
| 4 | `04_backtesting_strategy.ipynb` | Generates out-of-sample signals on the held-out 20% test period, computes CAGR / Sharpe / Drawdown / Calmar metrics, and visualises the equity curve vs buy-and-hold |

---

## 🔧 Features Engineered

All features are strictly backward-looking — zero look-ahead bias.

| Category | Features |
|----------|----------|
| Momentum | `ret_1m`, `ret_3m`, `ret_6m`, `ret_12m` |
| Volatility | `volatility_3m`, `volatility_6m` |
| Oscillator | `rsi_14` |
| Trend | `ma_ratio_3`, `ma_ratio_12` |
| Risk | `drawdown` (from 12m rolling high) |
| Range | `hl_ratio` (monthly high-low / close) |
| Volume | `volume_change` (3m pct change) |
| Seasonality | `month`, `quarter` |

---

## ⚙️ Methodology & Key Design Decisions

### Walk-Forward Cross-Validation
Standard K-Fold CV is invalid for financial time series — it can train on future data and evaluate on the past. All CV in this project uses `TimeSeriesSplit`, which strictly preserves temporal order.

### Nested Cross-Validation (Notebook 3)
The original approach of tuning hyperparameters with `RandomizedSearchCV` and then reporting that same CV score is biased — the selected parameters were chosen to maximise performance on those exact splits. Nested CV separates the two concerns:
- **Inner loop:** selects hyperparameters on training folds only
- **Outer loop:** evaluates the tuned model on a completely held-out fold

The reported outer AUC is an unbiased estimate of generalisation performance.

### Leak-Free Feature Importances (Notebook 2)
Feature importances are computed as the **mean across walk-forward folds**, not by retraining on the full dataset. Retraining on all data to extract importances leaks future price information into the importance estimates.

### Correct Commission Accounting (Notebook 4)
A round-trip trade costs **0.20%** total — 0.10% on entry and 0.10% on exit. Each leg is charged separately at every signal change.

### Train/Test Split
Both saved models are trained on the **first 80% of data only**. The last 20% (approximately mid-2022 to December 2024) is never touched during training or tuning and serves as the true out-of-sample test for the backtest.

---

## 📈 Results Summary

> Results correspond to the out-of-sample test period (~mid-2022 to December 2024).

| Metric | Buy & Hold BTC | GBM Strategy |
|--------|----------------|--------------|
| CAGR | — | — |
| Sharpe Ratio | — | — |
| Max Drawdown | — | — |
| Total Return | — | — |

*Run the notebooks to populate this table with your actual results.*

---

## ⚠️ Honest Limitations

The walk-forward AUC from Notebook 2 averages **0.424 across all five folds** — below the random baseline of 0.5. Only the most recent folds (covering the 2023–2024 bull run) show AUC above 0.5, suggesting the model captures the latest market regime but does not generalise robustly across different conditions. The strong backtest performance is partly a consequence of the test period coinciding with one of Bitcoin's clearest uptrends.

Critically, the sharp **Bitcoin correction of January–February 2025** — approximately a 25% drop from near $100,000 driven by macroeconomic and geopolitical factors — falls **entirely outside the dataset**, which ends in December 2024. This event had no prior technical signature in the features used here, and the model would almost certainly have been long entering January 2025, absorbing the full drawdown. This is an honest limitation of any price-only technical model facing a macro-driven structural break.

The methodology is rigorous; the results are promising but regime-dependent. That distinction matters more than any single backtest number.

---

## 🚀 Getting Started

### Requirements

```bash
pip install yfinance pandas numpy matplotlib seaborn scikit-learn lightgbm joblib scipy
```

### Run in order

```bash
jupyter notebook notebooks/01_data_eda.ipynb
jupyter notebook notebooks/02_gbm_model.ipynb
jupyter notebook notebooks/03_hyperparameter_tuning.ipynb
jupyter notebook notebooks/04_backtesting_strategy.ipynb
```

Each notebook saves its outputs to `data/` and `models/`, which are loaded by the subsequent notebook.

---

## 📄 Disclaimer

This project is for **educational and research purposes only**. Past performance does not guarantee future results. Nothing here constitutes investment advice.
