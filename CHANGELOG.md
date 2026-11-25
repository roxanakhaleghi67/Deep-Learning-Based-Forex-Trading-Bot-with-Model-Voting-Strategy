# Changelog

## Unreleased

### Added
- `reports/evaluate_results.py`: a standard-library evaluation script that writes
  `results_summary.md` and `equity_curve.svg` from `output.csv`.
- `utils/callbacks.py` with the `F1Metric` Keras callback used by training.
- Package markers (`__init__.py`) for `model`, `utils`, `strategy`, `backtest` and `tuning`.

### Fixed
- Broken imports: `models.model_architecture` → `model.model_architecture`, and the missing
  `F1Metric` symbol.
- `tuning/optuna_tuning.py` no longer starts a study when imported.
- `requirements.txt` pins versions; `MetaTrader5` is now documented as Windows-only and optional.
