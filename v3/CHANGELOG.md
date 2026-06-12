# Changelog

## [Unreleased] — 2026-04-08

### Added — Post-Training Analysis (`SnP500LSTMTransV3.py`)

#### New imports
- `confusion_matrix` from `sklearn.metrics`
- `scipy.stats` for skewness, kurtosis, and Q-Q plot support

#### New `HybridModelV3` methods

| Method | Description |
|---|---|
| `evaluate_split(X, y, split_name)` | Full evaluation on any data split. Returns regression metrics, error distribution stats, direction classification breakdown (precision/recall/TP/TN/FP/FN), prediction health diagnostics, and raw arrays for plotting. |
| `analyze_training_dynamics()` | Reads `self.history` to extract convergence epoch, overfitting gap/ratio, best val MAE and direction accuracy, learning rate trace, and a convergence stability measure (std of val loss over last 10 epochs). |
| `plot_error_analysis(results, save_path)` | 4-panel plot: residual histogram with fitted normal, Q-Q plot, residuals over time (shaded over/under), and `\|error\|` vs `\|actual return\|` scatter with trend line. |
| `plot_trading_simulation(results, save_path)` | Simulates a long/short strategy driven by direction predictions. Plots cumulative return vs buy-and-hold (with Sharpe and max drawdown in legend) and rolling direction accuracy. Returns a stats dict. |
| `plot_confusion_matrix_chart(results, save_path)` | Side-by-side raw count and row-normalised confusion matrices using seaborn heatmap. |
| `plot_split_comparison(split_results, save_path)` | Bar chart comparing MAE, RMSE, R², and direction accuracy across train / val / test splits. |

#### New standalone function

**`generate_full_report(model, data, save_path)`**

Orchestrates the entire post-training analysis pipeline and writes a human-readable report. Sections:

1. **Training Dynamics** — epochs, early stopping, overfitting gap/ratio, convergence verdict
2. **Training Split Performance** — regression + error distribution + direction metrics + collapse diagnostics
3. **Validation Split Performance** — same structure as section 2
4. **Test Set (Actual Data) Performance** — same structure as section 2
5. **Train vs Test Generalisation Summary** — side-by-side delta table for MAE, RMSE, R², direction accuracy
6. **Trading Simulation (Test Set)** — total return, annualised Sharpe, max drawdown vs buy-and-hold
7. **Feature Importance** — top-20 attention weights ranked
8. **Generated Output Files** — index of all saved plots and data files

Also saves `results/detailed_metrics_v3.json` with all metrics in machine-readable form.

#### New collapse / model health diagnostics (inside `evaluate_split`)

- `pred_std` / `actual_std` / `variance_ratio` — flags when model output has collapsed to near-constant
- `is_collapsed` — boolean, true when `variance_ratio < 0.05`
- `naive_mae`, `naive_dir_acc` — baseline scores for a predictor that always predicts the split mean
- `mae_skill_score` — how much better (%) the model is vs the naive baseline; negative = model is worse than predicting the mean

#### Model Health Summary banner in report

A `*** MODEL HEALTH SUMMARY ***` section is printed at the top of the report. When collapse is detected it:
- Warns explicitly that predictions are near-constant
- Shows predicted vs actual standard deviation
- Explains that direction accuracy is just reflecting market upward drift
- Lists the three most likely root causes (asymmetric loss local minimum, target scale mismatch, training hyperparameters)

#### Updated `main()`

- Replaced manual `evaluate()` + individual `plot_*` calls with a single `generate_full_report()` call
- Updated generated-files listing to reflect new outputs
- Return value simplified to `model` (metrics are now in `results/detailed_metrics_v3.json`)

### New output files produced per run

```
results/report_v3.txt                          # Full text report
results/detailed_metrics_v3.json               # Machine-readable metrics (all splits)
plots/error_analysis_train_v3.png
plots/error_analysis_val_v3.png
plots/error_analysis_test_v3.png
plots/trading_simulation_train_v3.png
plots/trading_simulation_val_v3.png
plots/trading_simulation_test_v3.png
plots/confusion_matrix_train_v3.png
plots/confusion_matrix_val_v3.png
plots/confusion_matrix_test_v3.png
plots/split_comparison_v3.png
```

### Root cause documented — model output collapse

The existing `predictions_v3.png` showed a flat red prediction line at ~+0.002. Root causes identified:

1. **Asymmetric loss** (`under_prediction_weight=2.0` in `biased_mse`) creates a local minimum where predicting a small constant positive value is never heavily penalised.
2. **Target scale mismatch** — `X` features are MinMaxScaler `[0,1]` but `y` targets are raw returns `~[-0.05, 0.05]`; the output head's linear layer stays near zero.
3. **Multiplicative growth bias** (`pred * (1 + 0.005)`) is a no-op near zero and does not inject a meaningful upward offset.

The new health diagnostics surface this automatically on every run.
