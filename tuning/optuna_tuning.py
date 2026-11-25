"""Optuna hyper-parameter search for the CNN-BiGRU + attention classifier.

The search needs an end-to-end data pipeline (load ``data.csv``, build
features, label, split, train, score). That pipeline is not wired yet, so
``train_evaluate_model`` raises a clear error instead of silently failing.
Importing this module has no side effects.
"""

from __future__ import annotations

import optuna

from model.model_architecture import build_classifier  # noqa: F401 - public re-export


def train_evaluate_model(params: dict) -> float:
    """Train with ``params`` and return the validation F1 score.

    Not implemented: requires the data pipeline described in ``ROADMAP.md``.
    """
    raise NotImplementedError(
        "train_evaluate_model is not implemented yet; "
        "see ROADMAP.md for the planned data pipeline."
    )


def objective(trial: optuna.Trial) -> float:
    params = {
        "filters": trial.suggest_int("filters", 64, 128),
        "units3": trial.suggest_int("units3", 32, 128),
        "units5": trial.suggest_int("units5", 64, 128),
        "dropout_rate2": trial.suggest_float("dropout_rate2", 0.1, 0.4),
        "dropout_rate3": trial.suggest_float("dropout_rate3", 0.1, 0.4),
        "dropout_rate5": trial.suggest_float("dropout_rate5", 0.1, 0.4),
        "kernel_size": trial.suggest_int("kernel_size", 3, 9),
        "gru_dropout1": trial.suggest_float("gru_dropout1", 0.1, 0.4),
        "num_heads": trial.suggest_int("num_heads", 4, 8),
        "key_dim": trial.suggest_int("key_dim", 8, 32),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2),
    }
    return train_evaluate_model(params)


def main() -> None:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)


if __name__ == "__main__":
    main()