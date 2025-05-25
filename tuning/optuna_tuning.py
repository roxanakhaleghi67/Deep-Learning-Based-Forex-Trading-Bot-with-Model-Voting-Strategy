import optuna
from models.model_architecture import build_classifier

def objective(trial):
    params = {
        'filters': trial.suggest_int("filters", 64, 128),
        'units3': trial.suggest_int("units3", 32, 128),
        'units5': trial.suggest_int("units5", 64, 128),
        'dropout_rate2': trial.suggest_float("dropout_rate2", 0.1, 0.4),
        'dropout_rate3': trial.suggest_float("dropout_rate3", 0.1, 0.4),
        'dropout_rate5': trial.suggest_float("dropout_rate5", 0.1, 0.4),
        'kernel_size': trial.suggest_int("kernel_size", 3, 9),
        'gru_dropout1': trial.suggest_float("gru_dropout1", 0.1, 0.4),
        'num_heads': trial.suggest_int("num_heads", 4, 8),
        'key_dim': trial.suggest_int("key_dim", 8, 32),
        'lr': trial.suggest_float("lr", 1e-4, 1e-2)
    }
    # باید یک تابع train_evaluate_model بنویسی که مدل رو آموزش بده و val_f1 برگردونه
    score = train_evaluate_model(params)
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
