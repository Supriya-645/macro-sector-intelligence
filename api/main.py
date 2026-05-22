"""
Phase 11 — FastAPI Backend.

Serves the AI predictions, backtest results, macro data, historical trends,
regimes, risk metrics, and simulation results to the new React frontend.
"""

import atexit
import asyncio
import json
import logging
import math
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_data_paths, MACRO_COLS
from src.sentiment import get_market_pulse

app = FastAPI(title="Macro Intelligence API")

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Daily email report scheduler
# ---------------------------------------------------------------------------
_scheduler = BackgroundScheduler()
_watcher_stop_event: threading.Event | None = None
_watcher_thread: threading.Thread | None = None


def _run_daily_report() -> None:
    """Scheduled job that sends the daily macro intelligence email report."""
    try:
        from src.email_report import send_email_report
        send_email_report([])
    except Exception as exc:
        logger.error("Daily email report failed: %s", exc)


_scheduler.add_job(_run_daily_report, "cron", hour=8, minute=0)


def _start_scheduler() -> None:
    """Start the background scheduler once."""
    if not _scheduler.running:
        _scheduler.start()


def _shutdown_scheduler() -> None:
    """Stop the background scheduler if it is active."""
    if _scheduler.running:
        _scheduler.shutdown(wait=False)


atexit.register(_shutdown_scheduler)

# WebSocket connection manager
_ws_clients: list[WebSocket] = []


@app.on_event("startup")
async def startup_background_tasks() -> None:
    """Start scheduled reports and dataset watcher on API startup."""
    global _watcher_stop_event, _watcher_thread

    _start_scheduler()

    if _watcher_thread and _watcher_thread.is_alive():
        return

    try:
        from api.watcher import watch_and_notify

        loop = asyncio.get_running_loop()
        watch_path = get_data_paths()["processed"] / "master_dataset.csv"
        _watcher_stop_event = threading.Event()
        _watcher_thread = threading.Thread(
            target=watch_and_notify,
            args=(_ws_clients, loop),
            kwargs={"watch_path": watch_path, "stop_event": _watcher_stop_event},
            daemon=True,
            name="macro-data-watcher",
        )
        _watcher_thread.start()
        logger.info("Started dataset watcher for %s", watch_path)
    except Exception as exc:
        logger.warning("Could not start dataset watcher: %s", exc)


@app.on_event("shutdown")
def shutdown_background_tasks() -> None:
    """Stop background tasks when the API shuts down."""
    if _watcher_stop_event:
        _watcher_stop_event.set()
    _shutdown_scheduler()


class SimulationInput(BaseModel):
    """Input parameters for simulating a macroeconomic shock."""
    US_Fed_Funds_Rate: float
    Brent_Crude: float
    DXY: float
    India_VIX: float


def clean_float(value):
    """Convert NaN or Infinity to None so JSON serialization works.

    Parameters
    ----------
    value : float
        The numerical value to clean.

    Returns
    -------
    float or None
        Cleaned float value suitable for JSON serialization.
    """
    if pd.isna(value) or math.isinf(value):
        return None
    return float(value)


@app.get("/api/overview")
def get_overview():
    """Returns the latest macro metrics and trend data."""
    paths = get_data_paths()
    try:
        master = pd.read_csv(paths["processed"] / "master_dataset.csv", index_col="Date", parse_dates=True)
    except Exception:
        return {"error": "Data not found"}

    if master.empty:
        return {"error": "Dataset is empty"}

    latest = master.iloc[-1]
    prev = master.iloc[-2] if len(master) > 1 else master.iloc[-1]

    metrics = []
    
    config = [
        ("Nifty_50", "Nifty 50 Index", True),
        ("India_VIX", "India VIX", False),
        ("US_Fed_Funds_Rate", "Fed Funds Rate (%)", False),
        ("Brent_Crude", "Brent Crude ($)", False),
        ("DXY", "US Dollar Index", False),
        ("US_10Y_Yield", "US 10Y Yield (%)", False),
    ]

    for col, title, is_higher_better in config:
        if col in master.columns:
            val = latest.get(col, np.nan)
            pval = prev.get(col, np.nan)
            
            if pd.isna(val):
                continue
                
            delta = val - pval
            pct_delta = (delta / pval * 100) if pval else 0

            # Get historical trend data for sparkline (last 24 months)
            trend_data = master[col].tail(24).dropna().tolist()

            metrics.append({
                "id": col,
                "title": title,
                "value": clean_float(val),
                "delta": clean_float(delta),
                "pct_delta": clean_float(pct_delta),
                "is_higher_better": is_higher_better,
                "trend": [clean_float(x) for x in trend_data]
            })

    return {"metrics": metrics}


@app.get("/api/predictions")
def get_predictions():
    """Returns next-month directional return forecasts for all sectors.

    Retrieves pre-trained XGBoost classifiers and applies them to the
    latest macroeconomic indicators to forecast directional probability.

    Returns
    -------
    dict
        List of predictions containing sector names, binary forecasts (1 for bullish,
        0 for bearish), model confidence, and top features driving the model.
    """
    paths = get_data_paths()
    import joblib
    
    try:
        master = pd.read_csv(paths["processed"] / "master_dataset.csv", index_col="Date", parse_dates=True)
        regime_df = pd.read_csv(paths["processed"] / "regime_labels.csv", index_col="Date", parse_dates=True)
    except Exception:
        return {"error": "Data not found"}

    models = {}
    if paths["models"].exists():
        for file in paths["models"].glob("xgb_*.joblib"):
            sector_name = file.stem.replace("xgb_", "")
            try:
                models[sector_name] = joblib.load(file)
            except Exception:
                pass

    if not models or master.empty:
        return {"predictions": []}

    avail_macros = [c for c in MACRO_COLS if c in master.columns]
    latest_data = pd.DataFrame(index=[master.index[-1]])

    for m in avail_macros:
        filled_macro = master[m].ffill().bfill()
        latest_data[m] = filled_macro.iloc[-1]
        latest_data[f"{m}_MoM"] = filled_macro.pct_change().iloc[-1]
        latest_data[f"{m}_YoY"] = filled_macro.pct_change(12).iloc[-1]

    if not regime_df.empty:
        latest_regime = regime_df["Regime"].iloc[-1]
        for r in ["Contraction", "Expansion", "Peak", "Recovery"]:
            latest_data[f"Regime_{r}"] = 1.0 if r == latest_regime else 0.0

    predictions = []
    
    for sector, model in models.items():
        try:
            model_features = model.feature_names_in_
            X_pred = latest_data.reindex(columns=model_features).fillna(0)
            prob = model.predict_proba(X_pred)[0]
            pred = model.predict(X_pred)[0]
            
            # Feature importance
            imp_df = pd.DataFrame({
                "Feature": model_features,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False).head(3)
            
            predictions.append({
                "sector": sector.replace("_Return", "").replace("_", " "),
                "prediction": int(pred),
                "confidence": clean_float(prob[pred]),
                "top_drivers": imp_df.to_dict(orient="records")
            })
        except Exception:
            continue

    return {"predictions": predictions}


@app.get("/api/backtest")
def get_backtest():
    """Returns the backtest results."""
    paths = get_data_paths()
    try:
        with open(paths["processed"] / "backtest_results.json") as f:
            data = json.load(f)
        return {"results": data}
    except Exception:
        return {"results": {}}


@app.get("/api/sentiment")
def get_sentiment():
    """Returns live news sentiment."""
    try:
        pulse = get_market_pulse()
        
        # Convert dataframe to dict, handling NaNs
        df = pulse.get("df", pd.DataFrame())
        if not df.empty:
            df = df.replace({np.nan: None})
            articles = df.to_dict(orient="records")
        else:
            articles = []
            
        return {
            "score": clean_float(pulse.get("score", 0.0)),
            "label": pulse.get("label", "Neutral"),
            "articles": articles
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/historical")
def get_historical():
    """Returns historical timeseries for all indicators and sectors."""
    paths = get_data_paths()
    try:
        master = pd.read_csv(paths["processed"] / "master_dataset.csv", parse_dates=["Date"])
        master["Date"] = master["Date"].dt.strftime("%Y-%m-%d")
        master = master.replace({np.nan: None})
        return {"data": master.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/regimes")
def get_regimes():
    """Returns the economic regime timeline and sector rotation matrix."""
    paths = get_data_paths()
    
    # Load K-Means timeline labels
    try:
        regime_df = pd.read_csv(paths["processed"] / "regime_labels.csv", parse_dates=["Date"])
        regime_df["Date"] = regime_df["Date"].dt.strftime("%Y-%m-%d")
        timeline = regime_df.to_dict(orient="records")
    except Exception:
        timeline = []

    # Load or compute HMM timeline labels. This keeps the historical
    # "timeline" key backward-compatible while exposing both model families.
    hmm_timeline = []
    hmm_error = None
    try:
        hmm_csv = paths["processed"] / "hmm_regime_labels.csv"
        if hmm_csv.exists():
            hmm_df = pd.read_csv(hmm_csv, parse_dates=["Date"])
        else:
            from src.hmm_regime import run_hmm_regime_detection
            hmm_df = run_hmm_regime_detection()
        if "HMM_Regime" in hmm_df.columns:
            hmm_df["Regime"] = hmm_df["HMM_Regime"]
        hmm_df["Date"] = hmm_df["Date"].dt.strftime("%Y-%m-%d")
        hmm_timeline = hmm_df.replace({np.nan: None}).to_dict(orient="records")
    except Exception as exc:
        hmm_error = str(exc)

    # Load rotation matrix (average returns in various environments)
    try:
        rotation_df = pd.read_csv(paths["processed"] / "sector_rotation_matrix.csv", index_col=0)
        rotation_data = {
            env: row.replace({np.nan: None}).to_dict()
            for env, row in rotation_df.iterrows()
        }
        sectors = [col.replace("_", " ") for col in rotation_df.columns]
        raw_sectors = list(rotation_df.columns)
        environments = list(rotation_df.index)
    except Exception:
        rotation_data = {}
        sectors = []
        raw_sectors = []
        environments = []

    return {
        "timeline": timeline,
        "timelines": {
            "kmeans": timeline,
            "hmm": hmm_timeline,
        },
        "hmm_error": hmm_error,
        "rotation": {
            "sectors": sectors,
            "raw_sectors": raw_sectors,
            "environments": environments,
            "matrix": rotation_data
        }
    }


@app.get("/api/risk")
def get_risk_metrics():
    """Returns annualized Sharpe ratio, Max Drawdown, and VaR table."""
    paths = get_data_paths()
    try:
        risk_df = pd.read_csv(paths["processed"] / "risk_metrics.csv")
        risk_df = risk_df.replace({np.nan: None})
        
        # Clean Sector and Regime names in response
        metrics = []
        for _, row in risk_df.iterrows():
            metrics.append({
                "sector": row["Sector"].replace("_", " "),
                "regime": row["Regime"],
                "sharpe_ratio": clean_float(row.get("Sharpe_Ratio")),
                "max_drawdown": clean_float(row.get("Max_Drawdown")),
                "var_95": clean_float(row.get("VaR_95")),
                "months": int(row.get("Months", 0)) if row.get("Months") is not None else 0
            })
        return {"metrics": metrics}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/simulate")
def run_simulation(inputs: SimulationInput):
    """Simulates sector returns under a hypothetical macroeconomic shock.

    Applies custom user slider values for Fed Rates, Brent Crude, DXY, and
    India VIX to recalculate XGBoost classification probabilities. Features MoM
    and YoY differences are derived based on latest historical points.

    Parameters
    ----------
    inputs : SimulationInput
        Pydantic model containing the custom macroeconomic levels.

    Returns
    -------
    dict
        Simulated probability of positive returns for each sector index.
    """
    paths = get_data_paths()
    import joblib

    try:
        master = pd.read_csv(paths["processed"] / "master_dataset.csv", index_col="Date", parse_dates=True)
        regime_df = pd.read_csv(paths["processed"] / "regime_labels.csv", index_col="Date", parse_dates=True)
    except Exception:
        return {"error": "Data not found"}

    models = {}
    if paths["models"].exists():
        for file in paths["models"].glob("xgb_*.joblib"):
            sector_name = file.stem.replace("xgb_", "")
            try:
                models[sector_name] = joblib.load(file)
            except Exception:
                pass

    if not models or master.empty:
        return {"error": "Models or dataset is empty"}

    latest = master.iloc[-1].copy()
    sim_data = pd.DataFrame(index=[0])

    avail_macros = [c for c in MACRO_COLS if c in master.columns]
    for m in avail_macros:
        if m == "US_Fed_Funds_Rate":
            val = inputs.US_Fed_Funds_Rate
        elif m == "Brent_Crude":
            val = inputs.Brent_Crude
        elif m == "DXY":
            val = inputs.DXY
        elif m == "India_VIX":
            val = inputs.India_VIX
        else:
            val = latest[m]

        sim_data[m] = val
        prev_val = master[m].iloc[-2] if len(master) > 1 else master[m].iloc[-1]
        sim_data[f"{m}_MoM"] = (val - prev_val) / prev_val if prev_val != 0 else 0

        if len(master) >= 12:
            yoy_val = master[m].iloc[-12]
            sim_data[f"{m}_YoY"] = (val - yoy_val) / yoy_val if yoy_val != 0 else 0
        else:
            sim_data[f"{m}_YoY"] = 0

    if not regime_df.empty:
        latest_regime = regime_df["Regime"].iloc[-1]
        for r in ["Contraction", "Expansion", "Peak", "Recovery"]:
            sim_data[f"Regime_{r}"] = 1.0 if r == latest_regime else 0.0

    results = []
    for sector, mdl in models.items():
        try:
            model_features = mdl.feature_names_in_
            X_sim = sim_data.reindex(columns=model_features).fillna(0)
            p = mdl.predict_proba(X_sim)[0][1]
            results.append({
                "sector": sector.replace("_Return", "").replace("_", " "),
                "probability": clean_float(p)
            })
        except Exception:
            continue

    return {"results": results}


@app.get("/api/tune/{sector}")
def tune_sector(sector: str):
    """Trigger Optuna hyper-parameter search for a given sector model.

    Parameters
    ----------
    sector : str
        The sector column name (e.g., 'Nifty_50_Return').

    Returns
    -------
    dict
        Best hyper-parameters and validation F1 score from the Optuna study.
    """
    try:
        from src.optuna_tuning import run_tuning
        result = run_tuning(sector, n_trials=30)
        return result
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/api/regimes/hmm")
def get_hmm_regimes():
    """Return HMM-based economic regime timeline.

    Loads cached HMM regime labels if available, otherwise runs
    the Hidden Markov Model detection pipeline on demand.

    Returns
    -------
    dict
        List of date-regime pairs from the HMM model.
    """
    paths = get_data_paths()
    hmm_csv = paths["processed"] / "hmm_regime_labels.csv"
    try:
        if hmm_csv.exists():
            df = pd.read_csv(hmm_csv, parse_dates=["Date"])
        else:
            from src.hmm_regime import run_hmm_regime_detection
            df = run_hmm_regime_detection()
        if "HMM_Regime" in df.columns:
            df["Regime"] = df["HMM_Regime"]
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        return {"timeline": df.replace({np.nan: None}).to_dict(orient="records")}
    except Exception as exc:
        return {"error": str(exc)}


@app.post("/api/trigger_email")
def trigger_email_report():
    """Manually trigger the daily macro intelligence email summary.

    Returns
    -------
    dict
        Status message indicating whether the email was dispatched.
    """
    try:
        from src.email_report import send_email_report
        success = send_email_report([])
        return {"status": "sent" if success else "failed"}
    except Exception as exc:
        return {"error": str(exc)}


@app.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket):
    """WebSocket endpoint that pushes live data-update notifications.

    Clients subscribe here to receive a ``{type: 'update'}`` message
    whenever the master dataset is refreshed.
    """
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            # Keep connection alive; broadcasting is driven by the watcher
            await websocket.receive_text()
    except WebSocketDisconnect:
        _ws_clients.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
