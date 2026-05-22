"""
HMM Regime Detection — Phase Enhancement 2.

Implements Hidden Markov Model based economic regime detection as an
alternative to K-Means clustering. Uses a GaussianHMM fitted on
engineered macro features to identify latent economic states.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import get_data_paths, MACRO_COLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Macro columns used for HMM feature engineering
HMM_MACRO_COLS = [
    "US_Fed_Funds_Rate",
    "US_CPI",
    "US_10Y_Yield",
    "Brent_Crude",
    "Gold_Price",
    "India_VIX",
    "USD_INR",
    "DXY",
]

# Regime labels ordered from worst to best average equity return
REGIME_ORDER = ["Contraction", "Recovery", "Expansion", "Peak"]


def _engineer_hmm_features(master: pd.DataFrame) -> pd.DataFrame:
    """Build macro feature matrix for HMM fitting.

    Engineers month-over-month change, year-over-year change, a
    6-month rolling z-score, and 3-month momentum for each macro
    indicator available in the dataset.

    Args:
        master: Master dataset DataFrame indexed by Date.

    Returns:
        Feature DataFrame with NaN rows dropped.
    """
    features = pd.DataFrame(index=master.index)
    avail = [c for c in HMM_MACRO_COLS if c in master.columns]

    for col in avail:
        s = master[col].ffill().bfill()
        features[f"{col}_MoM"] = s.pct_change()
        features[f"{col}_YoY"] = s.pct_change(12)
        roll_mean = s.rolling(6).mean()
        roll_std = s.rolling(6).std()
        features[f"{col}_Zscore"] = (s - roll_mean) / (roll_std + 1e-9)
        features[f"{col}_Mom3"] = s.diff(3)

    return features.dropna()


def run_hmm_regime_detection(n_states: int = 4) -> pd.DataFrame:
    """Fit a GaussianHMM on macro features and label economic regimes.

    Loads the master dataset, engineers features, standardizes them,
    fits a Gaussian HMM, then assigns human-readable regime labels
    by sorting states on their average Nifty 50 return (worst → best
    maps to Contraction → Recovery → Expansion → Peak).

    The result is saved to ``data/processed/hmm_regime_labels.csv``.

    Args:
        n_states: Number of hidden states in the HMM. Defaults to 4.

    Returns:
        DataFrame with columns: Date, HMM_Regime, HMM_State_ID.

    Raises:
        ImportError: If hmmlearn is not installed.
        FileNotFoundError: If master_dataset.csv is missing.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:
        raise ImportError(
            "hmmlearn is required. Install with: pip install hmmlearn"
        ) from exc

    paths = get_data_paths()
    master = pd.read_csv(
        paths["processed"] / "master_dataset.csv",
        index_col="Date",
        parse_dates=True,
    )

    features = _engineer_hmm_features(master)
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    logger.info("Fitting GaussianHMM with %d states on %d samples...", n_states, len(X))

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=42,
        verbose=False,
    )
    model.fit(X)
    state_ids = model.predict(X)

    result_df = pd.DataFrame({"HMM_State_ID": state_ids}, index=features.index)

    # Align Nifty 50 returns to assign meaningful labels
    if "Nifty_50_Return" in master.columns:
        result_df["Nifty_50_Return"] = master["Nifty_50_Return"].reindex(features.index)
        mean_returns = (
            result_df.groupby("HMM_State_ID")["Nifty_50_Return"]
            .mean()
            .sort_values()
        )
        state_to_label = {
            state: REGIME_ORDER[i] for i, state in enumerate(mean_returns.index)
        }
    else:
        state_to_label = {i: REGIME_ORDER[i % len(REGIME_ORDER)] for i in range(n_states)}

    result_df["HMM_Regime"] = result_df["HMM_State_ID"].map(state_to_label)
    result_df = result_df[["HMM_Regime", "HMM_State_ID"]].copy()
    result_df.index.name = "Date"
    result_df = result_df.reset_index()

    output_path = paths["processed"] / "hmm_regime_labels.csv"
    result_df.to_csv(output_path, index=False)
    logger.info("HMM regime labels saved to %s", output_path)

    state_counts = result_df["HMM_Regime"].value_counts().to_dict()
    logger.info("State distribution: %s", state_counts)

    return result_df


def load_hmm_labels() -> pd.DataFrame:
    """Load pre-computed HMM regime labels from disk.

    Args:
        None

    Returns:
        DataFrame with columns: Date, HMM_Regime, HMM_State_ID.
        Returns empty DataFrame if file is not found.
    """
    paths = get_data_paths()
    csv_path = paths["processed"] / "hmm_regime_labels.csv"

    if not csv_path.exists():
        logger.warning("HMM labels not found at %s. Run run_hmm_regime_detection() first.", csv_path)
        return pd.DataFrame()

    return pd.read_csv(csv_path, parse_dates=["Date"])


if __name__ == "__main__":
    df = run_hmm_regime_detection()
    print("\n=== HMM Regime Detection Complete ===")
    print(f"Total observations: {len(df)}")
    print("\nState distribution:")
    print(df["HMM_Regime"].value_counts())
    print("\nSample (tail):")
    print(df.tail(10).to_string(index=False))
