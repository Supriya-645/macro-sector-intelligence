"""
Phase 9 — VectorBT Strategy Backtester.

Simulates a directional trading strategy driven by the XGBoost classifier
predictions from Phase 5.  Strategy logic:

    - **Long** when the XGBoost model predicts UP (1).
    - **Flat / Cash** when the model predicts DOWN (0).

Generates equity curves, computes performance statistics, and benchmarks
the AI strategy against passive Buy-and-Hold.

Functions
---------
load_predictions    : Reconstruct walk-forward predictions for a sector.
run_backtest        : Execute the strategy simulation via vectorbt.
run_all_backtests   : Run backtests for every available model.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (
    load_master_dataset,
    load_regime_labels,
    setup_plotting_style,
    save_chart,
    get_data_paths,
    SECTOR_RETURN_COLS,
    MACRO_COLS,
    PALETTE,
)

# Remove Gold_Price if still in list (consistent with ml_models.py)
_macro_cols = [c for c in MACRO_COLS if c != "Gold_Price"]


# ---------------------------------------------------------------------------
# Prediction Reconstruction
# ---------------------------------------------------------------------------

def load_predictions(
    sector_return_col: str,
    df: pd.DataFrame,
    regime_df: pd.DataFrame,
    initial_train_pct: float = 0.6,
) -> pd.DataFrame:
    """Reconstruct walk-forward XGBoost predictions for a sector.

    Re-runs the walk-forward loop from Phase 5 to obtain dated
    predictions.  This avoids needing to persist predictions on disk.

    Parameters
    ----------
    sector_return_col : str
        Column name in ``df`` for the sector's monthly return.
    df : pandas.DataFrame
        Master dataset with DatetimeIndex.
    regime_df : pandas.DataFrame
        Regime labels with DatetimeIndex.
    initial_train_pct : float
        Fraction of data used for the initial training window.

    Returns
    -------
    pandas.DataFrame
        Indexed by Date with columns ``prediction`` (0/1) and
        ``actual_return`` (float).
    """
    import xgboost as xgb

    # Build features (same logic as ml_models.py)
    avail_macros = [c for c in _macro_cols if c in df.columns]

    features = pd.DataFrame(index=df.index)
    for m in avail_macros:
        filled = df[m].ffill().bfill()
        features[m] = filled
        features[f"{m}_MoM"] = filled.pct_change()
        features[f"{m}_YoY"] = filled.pct_change(12)

    if not regime_df.empty and "Regime" in regime_df.columns:
        dummies = pd.get_dummies(regime_df["Regime"], prefix="Regime").astype(float)
        features = features.join(dummies)

    target = (df[sector_return_col].shift(-1) > 0).astype(int)
    target.name = "Target"

    combined = pd.concat([features, target, df[sector_return_col]], axis=1).dropna()

    X = combined.drop(["Target", sector_return_col], axis=1)
    y = combined["Target"]
    returns = combined[sector_return_col]

    n = len(X)
    split = int(n * initial_train_pct)

    if n < 60 or split < 30:
        return pd.DataFrame()

    preds = []
    for i in range(split, n):
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X.iloc[:i], y.iloc[:i])
        pred = model.predict(X.iloc[[i]])[0]
        preds.append({
            "Date": X.index[i],
            "prediction": pred,
            "actual_return": returns.iloc[i],
        })

    return pd.DataFrame(preds).set_index("Date")


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def run_backtest(
    predictions_df: pd.DataFrame,
    sector_name: str,
) -> Optional[Dict]:
    """Run vectorbt backtest on a prediction DataFrame.

    Strategy: go long the sector when prediction == 1, stay in cash
    when prediction == 0.

    Parameters
    ----------
    predictions_df : pandas.DataFrame
        Must have ``prediction`` (0/1) and ``actual_return`` columns.
    sector_name : str
        Human-readable sector name for labelling charts.

    Returns
    -------
    dict or None
        Performance statistics including cumulative return, win rate,
        max drawdown, and Sharpe ratio.  Returns ``None`` on failure.
    """
    try:
        import vectorbt as vbt
    except ImportError:
        print("   vectorbt not installed. Using manual backtest.")
        return _manual_backtest(predictions_df, sector_name)

    if predictions_df.empty:
        return None

    returns = predictions_df["actual_return"]
    signals = predictions_df["prediction"]

    # Strategy returns: actual_return when signal == 1, else 0
    strategy_returns = returns * signals
    bh_returns = returns

    # Build cumulative equity curves
    strategy_equity = (1 + strategy_returns).cumprod()
    bh_equity = (1 + bh_returns).cumprod()

    # Metrics
    total_trades = int(signals.sum())
    win_trades = int(((strategy_returns > 0) & (signals == 1)).sum())
    win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0

    strategy_cum = strategy_equity.iloc[-1] - 1
    bh_cum = bh_equity.iloc[-1] - 1

    # Max drawdown
    peak = strategy_equity.cummax()
    dd = (strategy_equity - peak) / peak
    max_dd = dd.min()

    # Annualised Sharpe (monthly data)
    if strategy_returns.std() != 0:
        sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(12)
    else:
        sharpe = 0.0

    stats = {
        "sector": sector_name,
        "strategy_return": float(strategy_cum),
        "buyhold_return": float(bh_cum),
        "win_rate": float(win_rate),
        "max_drawdown": float(max_dd),
        "sharpe_ratio": float(sharpe),
        "total_signals": total_trades,
    }

    # Plot equity curves
    _plot_equity(strategy_equity, bh_equity, sector_name, stats)

    return stats


def _manual_backtest(
    predictions_df: pd.DataFrame,
    sector_name: str,
) -> Optional[Dict]:
    """Fallback manual backtest if vectorbt is unavailable.

    Uses the same long-when-predicted-up logic but without
    the vectorbt library.

    Parameters
    ----------
    predictions_df : pandas.DataFrame
        Must have ``prediction`` and ``actual_return`` columns.
    sector_name : str
        Human-readable name for charting.

    Returns
    -------
    dict or None
        Same structure as :func:`run_backtest`.
    """
    if predictions_df.empty:
        return None

    returns = predictions_df["actual_return"]
    signals = predictions_df["prediction"]

    strategy_returns = returns * signals
    bh_returns = returns

    strategy_equity = (1 + strategy_returns).cumprod()
    bh_equity = (1 + bh_returns).cumprod()

    total_trades = int(signals.sum())
    win_trades = int(((strategy_returns > 0) & (signals == 1)).sum())
    win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0

    strategy_cum = strategy_equity.iloc[-1] - 1
    bh_cum = bh_equity.iloc[-1] - 1

    peak = strategy_equity.cummax()
    dd = (strategy_equity - peak) / peak
    max_dd = dd.min()

    if strategy_returns.std() != 0:
        sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(12)
    else:
        sharpe = 0.0

    stats = {
        "sector": sector_name,
        "strategy_return": float(strategy_cum),
        "buyhold_return": float(bh_cum),
        "win_rate": float(win_rate),
        "max_drawdown": float(max_dd),
        "sharpe_ratio": float(sharpe),
        "total_signals": total_trades,
    }

    _plot_equity(strategy_equity, bh_equity, sector_name, stats)
    return stats


def _plot_equity(
    strategy_equity: pd.Series,
    bh_equity: pd.Series,
    sector_name: str,
    stats: Dict,
) -> None:
    """Plot strategy vs buy-and-hold equity curves.

    Parameters
    ----------
    strategy_equity : pandas.Series
        Cumulative strategy equity.
    bh_equity : pandas.Series
        Cumulative buy-and-hold equity.
    sector_name : str
        Sector name for chart title.
    stats : dict
        Performance statistics for annotation.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        strategy_equity.index,
        strategy_equity.values,
        label=f"AI Strategy ({stats['strategy_return']:+.1%})",
        color=PALETTE["accent_green"],
        linewidth=2.5,
    )
    ax.plot(
        bh_equity.index,
        bh_equity.values,
        label=f"Buy & Hold ({stats['buyhold_return']:+.1%})",
        color=PALETTE["accent_blue"],
        linewidth=2,
        linestyle="--",
    )

    ax.fill_between(
        strategy_equity.index,
        strategy_equity.values,
        1,
        where=strategy_equity.values >= 1,
        alpha=0.08,
        color=PALETTE["accent_green"],
    )

    ax.set_title(
        f"Backtest: AI Strategy vs Buy & Hold — {sector_name}",
        fontsize=16,
        pad=20,
    )
    ax.set_ylabel("Portfolio Value (₹1 = Start)", fontsize=12)
    ax.legend(fontsize=12, loc="upper left")

    # Annotation box
    textstr = (
        f"Win Rate: {stats['win_rate']:.1f}%\n"
        f"Max DD: {stats['max_drawdown']:.1%}\n"
        f"Sharpe: {stats['sharpe_ratio']:.2f}"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor=PALETTE["bg_card"], alpha=0.9,
                 edgecolor=PALETTE["grid"])
    ax.text(
        0.98, 0.02, textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
        color=PALETTE["text"],
    )

    save_chart(fig, f"backtest_{sector_name}", subfolder="backtest")


# ---------------------------------------------------------------------------
# Top-Level Runner
# ---------------------------------------------------------------------------

def run_all_backtests() -> Dict:
    """Execute backtests for all sectors with saved XGBoost models.

    Loads models from disk, regenerates walk-forward predictions,
    and runs the backtest for each.

    Returns
    -------
    dict
        Mapping of sector name to performance statistics dict.
    """
    print("=" * 60)
    print("  PHASE 9a — STRATEGY BACKTESTER")
    print("=" * 60)

    setup_plotting_style()
    df = load_master_dataset()

    try:
        regime_df = load_regime_labels()
    except Exception:
        regime_df = pd.DataFrame()

    paths = get_data_paths()
    models_dir = paths["models"]

    if not models_dir.exists():
        print("  No models found. Run Phase 5 first.")
        return {}

    model_files = list(models_dir.glob("xgb_*.joblib"))
    if not model_files:
        print("  No XGBoost models found in models directory.")
        return {}

    all_stats = {}

    for mf in model_files:
        sector_col = mf.stem.replace("xgb_", "")
        clean_name = sector_col.replace("_Return", "")
        print(f"  Backtesting {clean_name}...")

        if sector_col not in df.columns:
            print(f"     {sector_col} not in dataset. Skipping.")
            continue

        preds_df = load_predictions(sector_col, df, regime_df)

        if preds_df.empty:
            print(f"     Not enough data for {clean_name}.")
            continue

        stats = run_backtest(preds_df, clean_name)

        if stats:
            all_stats[clean_name] = stats
            print(
                f"    {clean_name}: "
                f"AI {stats['strategy_return']:+.1%} vs "
                f"B&H {stats['buyhold_return']:+.1%} | "
                f"Win Rate {stats['win_rate']:.0f}%"
            )

    # Save summary
    if all_stats:
        summary_path = paths["processed"] / "backtest_results.json"
        with open(summary_path, "w") as f:
            json.dump(all_stats, f, indent=4)
        print(f"\n  Saved backtest results to {summary_path.name}")

    print("=" * 60)
    print("  BACKTESTING COMPLETE")
    print("=" * 60)

    return all_stats


if __name__ == "__main__":
    run_all_backtests()
