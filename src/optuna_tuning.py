"""
Optuna Hyper-Parameter Tuning — Phase Enhancement 1.

Provides automated hyper-parameter search for the XGBoost sector
classification models using Optuna's TPE sampler and walk-forward
cross-validation to avoid look-ahead bias.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from src.utils import get_data_paths, MACRO_COLS, SECTOR_RETURN_COLS

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _engineer_features(
    master: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> pd.DataFrame:
    """Engineer lagged macro features and one-hot regime dummies.

    Args:
        master: Master dataset with macro indicators and sector returns.
        regime_df: DataFrame containing 'Regime' column indexed by Date.

    Returns:
        Feature DataFrame aligned with the master index, with NaN rows
        dropped at the head.
    """
    avail_macros = [c for c in MACRO_COLS if c in master.columns]
    features = pd.DataFrame(index=master.index)

    for col in avail_macros:
        series = master[col].ffill().bfill()
        features[col] = series
        features[f"{col}_MoM"] = series.pct_change()
        features[f"{col}_YoY"] = series.pct_change(12)

    if not regime_df.empty and "Regime" in regime_df.columns:
        aligned = regime_df["Regime"].reindex(master.index, method="ffill")
        for label in ["Contraction", "Expansion", "Peak", "Recovery"]:
            features[f"Regime_{label}"] = (aligned == label).astype(float)

    return features.dropna()


def _objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 3,
) -> float:
    """Optuna objective function using walk-forward cross-validation.

    Args:
        trial: Optuna trial object used to suggest hyper-parameters.
        X: Feature matrix.
        y: Binary target array.
        n_splits: Number of time-series cross-validation folds.

    Returns:
        Mean F1 score across all CV folds (macro-averaged).
    """
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
        "verbosity": 0,
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if len(np.unique(y_train)) < 2:
            continue

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds, average="macro", zero_division=0))

    return float(np.mean(scores)) if scores else 0.0


def run_tuning(sector_name: str, n_trials: int = 50) -> dict[str, Any]:
    """Run Optuna hyper-parameter tuning for a given sector model.

    Loads the master dataset and regime labels, engineers features,
    creates a binary directional return target, and runs an Optuna
    study using walk-forward cross-validation. The best parameters
    are persisted to ``data/processed/models/optuna_params.json``.

    Args:
        sector_name: Column name of the sector return to model
            (e.g., 'Nifty_50_Return').
        n_trials: Number of Optuna trials to run. Defaults to 50.

    Returns:
        Dictionary with keys:
            - ``sector``: Sector name.
            - ``best_params``: Best hyper-parameter dictionary.
            - ``best_score``: Mean F1 score achieved.
            - ``n_trials``: Number of trials run.
    """
    paths = get_data_paths()

    master = pd.read_csv(
        paths["processed"] / "master_dataset.csv",
        index_col="Date",
        parse_dates=True,
    )

    try:
        regime_df = pd.read_csv(
            paths["processed"] / "regime_labels.csv",
            index_col="Date",
            parse_dates=True,
        )
    except FileNotFoundError:
        regime_df = pd.DataFrame()

    if sector_name not in master.columns:
        # Try with _Return suffix
        sector_name = sector_name if sector_name.endswith("_Return") else f"{sector_name}_Return"

    if sector_name not in master.columns:
        raise ValueError(f"Sector '{sector_name}' not found in dataset.")

    features = _engineer_features(master, regime_df)
    target = (master[sector_name].shift(-1) > 0).astype(int)

    aligned_idx = features.index.intersection(target.dropna().index)
    X = features.loc[aligned_idx].values
    y = target.loc[aligned_idx].values

    logger.info("Running Optuna tuning for %s with %d trials...", sector_name, n_trials)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: _objective(t, X, y), n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value

    # Persist best params
    params_path = paths["models"] / "optuna_params.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if params_path.exists():
        with open(params_path) as f:
            existing = json.load(f)

    existing[sector_name] = {"params": best_params, "best_f1": best_score}

    with open(params_path, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info("Best F1 for %s: %.4f", sector_name, best_score)

    return {
        "sector": sector_name,
        "best_params": best_params,
        "best_score": round(best_score, 4),
        "n_trials": n_trials,
    }


def load_best_params(sector_name: str) -> dict[str, Any]:
    """Load the best Optuna hyper-parameters for a sector model.

    Args:
        sector_name: Column name of the sector (e.g., 'Nifty_50_Return').

    Returns:
        Dictionary of best XGBoost hyper-parameters, or empty dict if
        no tuning results exist for this sector.
    """
    paths = get_data_paths()
    params_path = paths["models"] / "optuna_params.json"

    if not params_path.exists():
        return {}

    with open(params_path) as f:
        all_params = json.load(f)

    return all_params.get(sector_name, {}).get("params", {})


if __name__ == "__main__":
    result = run_tuning("Nifty_50_Return", n_trials=20)
    print(f"\n=== Optuna Tuning Complete ===")
    print(f"Sector  : {result['sector']}")
    print(f"Best F1 : {result['best_score']}")
    print(f"Params  : {result['best_params']}")
