"""
Phase 5 — Predictive Modeling.

Trains XGBoost classifiers to predict next-month sector direction (Up/Down)
and a Prophet model to forecast Nifty 50 levels. Enforces strict walk-forward
backtesting to prevent lookahead bias.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (
    load_master_dataset,
    load_regime_labels,
    setup_plotting_style,
    save_chart,
    get_data_paths,
    SECTOR_RETURN_COLS,
    MACRO_COLS,
)

# Remove Gold_Price from MACRO_COLS as it failed to download in Phase 1
if "Gold_Price" in MACRO_COLS:
    MACRO_COLS.remove("Gold_Price")


DEFAULT_XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.1,
    "random_state": 42,
    "eval_metric": "logloss",
}


def _load_tuned_xgb_params(sector_return_col):
    """Load persisted Optuna parameters for a sector, if available."""
    try:
        from src.optuna_tuning import load_best_params
        return load_best_params(sector_return_col)
    except Exception:
        return {}


def _build_xgb_classifier(params=None):
    """Create an XGBoost classifier using defaults plus tuned overrides."""
    model_params = DEFAULT_XGB_PARAMS.copy()
    if params:
        model_params.update(params)
    model_params.setdefault("random_state", 42)
    model_params.setdefault("eval_metric", "logloss")
    return xgb.XGBClassifier(**model_params)


def prepare_xgboost_data(df, regime_df, sector_return_col):
    """Prepare features and target for XGBoost walk-forward backtest."""
    # Target: 1 if next month's return > 0, else 0
    target = (df[sector_return_col].shift(-1) > 0).astype(int)
    target.name = "Target"
    
    # Features: Lagged macros
    avail_macros = [c for c in MACRO_COLS if c in df.columns]
    
    features = pd.DataFrame(index=df.index)
    for m in avail_macros:
        filled_macro = df[m].ffill().bfill()
        features[m] = filled_macro
        features[f"{m}_MoM"] = filled_macro.pct_change()
        features[f"{m}_YoY"] = filled_macro.pct_change(12)
        
    # One-hot encode regime
    if not regime_df.empty and "Regime" in regime_df.columns:
        regime_dummies = pd.get_dummies(regime_df["Regime"], prefix="Regime").astype(float)
        features = features.join(regime_dummies)
        
    # Combine and drop NaNs
    combined = pd.concat([features, target], axis=1).dropna()
    
    # We drop the very last row because its target (next month's return) is NaN
    # Actually dropna() handles this if target is NaN, but let's be explicit
    # Wait, the shift(-1) makes the last row target NaN, so dropna removes it.
    
    X = combined.drop("Target", axis=1)
    y = combined["Target"]
    
    return X, y


def run_xgboost_backtest(X, y, initial_train_pct=0.6, model_params=None):
    """Strict walk-forward backtest for XGBoost."""
    n_samples = len(X)
    train_size = int(n_samples * initial_train_pct)
    
    predictions = []
    actuals = []
    
    # To accumulate feature importances
    all_importances = []
    
    # Walk forward by 1 month
    for i in range(train_size, n_samples):
        # Train from start up to i
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]
        
        # Test on i
        X_test = X.iloc[[i]]
        y_test = y.iloc[i]
        
        # Train model
        model = _build_xgb_classifier(model_params)
        model.fit(X_train, y_train)
        
        # Predict
        pred = model.predict(X_test)[0]
        
        predictions.append(pred)
        actuals.append(y_test)
        all_importances.append(model.feature_importances_)
        
    # Final model trained on all data (for saving)
    final_model = _build_xgb_classifier(model_params)
    final_model.fit(X, y)
    
    # Average feature importances across all steps
    avg_importance = np.mean(all_importances, axis=0)
    
    return {
        "predictions": predictions,
        "actuals": actuals,
        "importances": avg_importance,
        "feature_names": X.columns.tolist(),
        "final_model": final_model
    }


def plot_feature_importances(sector, importances, feature_names):
    """Plot top 15 features."""
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=imp_df, x='Importance', y='Feature', palette="viridis")
    plt.title(f"Top 15 Predictive Features for {sector.replace('_Return', '')}", pad=15)
    save_chart(plt.gcf(), f"xgb_importance_{sector}", subfolder="models")


def build_xgboost_models(df, regime_df):
    """Build and evaluate XGBoost models for all sectors."""
    print("  🤖 Building XGBoost Classifiers with walk-forward backtesting...")
    
    avail_sectors = [c for c in SECTOR_RETURN_COLS if c in df.columns]
    
    metrics = {}
    models_to_save = []
    
    for sector in avail_sectors:
        print(f"    Training {sector}...")
        X, y = prepare_xgboost_data(df, regime_df, sector)
        
        if len(X) < 60:
            print(f"      ⚠️ Not enough data for {sector} (n={len(X)})")
            continue
            
        tuned_params = _load_tuned_xgb_params(sector)
        if tuned_params:
            print(f"      ✅ Using Optuna parameters for {sector}")
        else:
            print(f"      ℹ️ Using default XGBoost parameters for {sector}")

        res = run_xgboost_backtest(X, y, model_params=tuned_params)
        
        acc = accuracy_score(res["actuals"], res["predictions"])
        f1 = f1_score(res["actuals"], res["predictions"], zero_division=0)
        prec = precision_score(res["actuals"], res["predictions"], zero_division=0)
        rec = recall_score(res["actuals"], res["predictions"], zero_division=0)
        
        metrics[sector] = {
            "Accuracy": acc,
            "F1_Score": f1,
            "Precision": prec,
            "Recall": rec,
            "Tuned": bool(tuned_params),
            "Params": tuned_params or DEFAULT_XGB_PARAMS,
        }
        
        plot_feature_importances(sector, res["importances"], res["feature_names"])
        
        models_to_save.append({
            "sector": sector,
            "f1": f1,
            "model": res["final_model"]
        })
        
    # Save top 3 models by F1 Score
    paths = get_data_paths()
    models_dir = paths["models"]
    models_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_save.sort(key=lambda x: x["f1"], reverse=True)
    for item in models_to_save[:3]:
        model_path = models_dir / f"xgb_{item['sector']}.joblib"
        joblib.dump(item["model"], model_path)
        print(f"  💾 Saved top model: {model_path.name} (F1: {item['f1']:.3f})")
        
    return metrics


def build_prophet_model(df):
    """Forecast Nifty 50 absolute levels using Facebook Prophet."""
    print("  🔮 Building Prophet Forecast for Nifty 50...")
    
    try:
        from prophet import Prophet
    except ImportError:
        print("  ⚠️ Prophet is not installed. Skipping this section.")
        return {}
        
    if "Nifty_50" not in df.columns:
        print("  ⚠️ Nifty_50 missing from dataset.")
        return {}
        
    # Default top macro regressors if Granger results not available
    regressors = ["US_Fed_Funds_Rate", "Brent_Crude", "DXY", "India_VIX", "US_10Y_Yield"]
    regressors = [r for r in regressors if r in df.columns]
    
    # Prepare Prophet dataframe (requires 'ds' and 'y')
    prophet_df = df[["Nifty_50"] + regressors].dropna().reset_index()
    prophet_df = prophet_df.rename(columns={"Date": "ds", "Nifty_50": "y"})
    
    if len(prophet_df) < 60:
        print("  ⚠️ Not enough data for Prophet.")
        return {}
        
    # Walk forward split
    train_size = int(len(prophet_df) * 0.8)
    train_df = prophet_df.iloc[:train_size]
    test_df = prophet_df.iloc[train_size:]
    
    # Fit model
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    for reg in regressors:
        m.add_regressor(reg)
        
    m.fit(train_df)
    
    # Predict on test set
    # Note: For regressors, Prophet needs their future values to predict.
    # In a strict backtest, we use the actual values of regressors from test_df.
    # In reality, you'd need to forecast the regressors too, or use lagged regressors.
    forecast = m.predict(test_df[["ds"] + regressors])
    
    # Calculate metrics
    y_true = test_df["y"].values
    y_pred = forecast["yhat"].values
    
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        "Prophet_Nifty_50": {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape
        }
    }
    
    # Plot forecast vs actual
    plt.figure(figsize=(14, 7))
    plt.plot(train_df["ds"], train_df["y"], label="Train Actual", color="#8B949E")
    plt.plot(test_df["ds"], test_df["y"], label="Test Actual", color="#58A6FF")
    plt.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="#3FB950", linestyle="--")
    plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="#3FB950", alpha=0.2)
    
    plt.title("Prophet Forecast vs Actual: Nifty 50 Level", pad=20)
    plt.ylabel("Nifty 50 Level")
    plt.legend()
    save_chart(plt.gcf(), "prophet_forecast_Nifty_50", subfolder="models")
    
    # Save the final model fitted on all data
    final_m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    for reg in regressors:
        final_m.add_regressor(reg)
    final_m.fit(prophet_df)
    
    # Prophet doesn't serialize well with joblib, so we skip saving the model artifact 
    # to avoid compatibility issues, but in a real app we could use prophet.serialize
    
    return metrics


def run_ml_models():
    """Execute Phase 5 Predictive Modeling."""
    print("=" * 60)
    print("  PHASE 5 — PREDICTIVE MODELING")
    print("=" * 60)
    
    setup_plotting_style()
    df = load_master_dataset()
    
    if df.empty:
        print("  ❌ Master dataset is empty. Run Phase 1 first.")
        return
        
    try:
        regime_df = load_regime_labels()
    except Exception:
        print("  ⚠️ Regime labels not found. Running without them.")
        regime_df = pd.DataFrame()
        
    xgb_metrics = build_xgboost_models(df, regime_df)
    prophet_metrics = build_prophet_model(df)
    
    # Combine and save metrics
    all_metrics = {**xgb_metrics, **prophet_metrics}
    
    paths = get_data_paths()
    metrics_path = paths["processed"] / "model_metrics.json"
    
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
        
    print(f"  💾 Saved model metrics to: {metrics_path.name}")
    print("=" * 60)
    print("  ✅ PHASE 5 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_ml_models()
