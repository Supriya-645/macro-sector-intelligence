"""
Pytest fixtures shared across all test modules.

Provides sample DataFrames and a temporary data directory so tests
run in isolation without touching production files.
"""

import shutil
import tempfile
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_master_df() -> pd.DataFrame:
    """Return a minimal 24-row master dataset for testing.

    Returns:
        DataFrame with Date index and representative macro + sector columns.
    """
    dates = pd.date_range("2008-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(seed=42)

    return pd.DataFrame(
        {
            "Date": dates,
            "Nifty_50": 5000 + rng.normal(0, 300, 24).cumsum(),
            "US_Fed_Funds_Rate": np.linspace(4.5, 0.25, 24),
            "US_CPI": np.linspace(3.5, 1.5, 24) + rng.normal(0, 0.1, 24),
            "US_10Y_Yield": np.linspace(4.0, 2.5, 24) + rng.normal(0, 0.2, 24),
            "Brent_Crude": 80 + rng.normal(0, 15, 24).cumsum() * 0.3,
            "Gold_Price": 900 + rng.normal(0, 30, 24).cumsum(),
            "India_VIX": 18 + rng.normal(0, 5, 24),
            "USD_INR": 45 + rng.normal(0, 1, 24).cumsum() * 0.2,
            "DXY": 80 + rng.normal(0, 3, 24).cumsum() * 0.3,
            "Global_VIX": 22 + rng.normal(0, 6, 24),
            "India_GDP": rng.normal(7, 1.5, 24),
            "US_GDP": rng.normal(2, 0.5, 24),
            "Nifty_50_Return": rng.normal(0.005, 0.04, 24),
            "Nifty_Bank_Return": rng.normal(0.006, 0.05, 24),
            "Nifty_IT_Return": rng.normal(0.008, 0.05, 24),
        }
    )


@pytest.fixture
def sample_regime_df() -> pd.DataFrame:
    """Return a minimal 24-row regime labels DataFrame for testing.

    Returns:
        DataFrame with Date and Regime columns cycling through
        the four economic states.
    """
    dates = pd.date_range("2008-01-01", periods=24, freq="MS")
    labels = ["Expansion", "Peak", "Contraction", "Recovery"]
    return pd.DataFrame(
        {
            "Date": dates,
            "Regime": [labels[i % 4] for i in range(24)],
        }
    )


@pytest.fixture(scope="session")
def tmp_data_dir():
    """Create a temporary data directory tree populated with sample CSVs.

    Yields:
        Path to the temporary processed data directory.

    Cleanup:
        The entire temp directory is removed after the test session.
    """
    tmpdir = Path(tempfile.mkdtemp())
    processed = tmpdir / "processed"
    models = processed / "models"
    processed.mkdir(parents=True)
    models.mkdir(parents=True)

    # Write sample master dataset
    rng = np.random.default_rng(seed=0)
    dates = pd.date_range("2008-01-01", periods=36, freq="MS")
    master = pd.DataFrame(
        {
            "Date": dates,
            "Nifty_50": 5000 + rng.normal(0, 200, 36).cumsum(),
            "US_Fed_Funds_Rate": np.linspace(4.5, 0.25, 36),
            "US_CPI": np.linspace(3.5, 1.5, 36),
            "US_10Y_Yield": np.linspace(4.0, 2.5, 36),
            "Brent_Crude": 75 + rng.normal(0, 8, 36).cumsum() * 0.2,
            "Gold_Price": 900 + rng.normal(0, 20, 36).cumsum(),
            "India_VIX": np.abs(18 + rng.normal(0, 4, 36)),
            "USD_INR": 45 + rng.normal(0, 0.5, 36).cumsum(),
            "DXY": 80 + rng.normal(0, 2, 36).cumsum() * 0.2,
            "Global_VIX": np.abs(22 + rng.normal(0, 5, 36)),
            "India_GDP": rng.normal(7, 1.5, 36),
            "US_GDP": rng.normal(2, 0.5, 36),
            "Nifty_50_Return": rng.normal(0.005, 0.04, 36),
            "Nifty_Bank_Return": rng.normal(0.006, 0.05, 36),
            "Nifty_IT_Return": rng.normal(0.008, 0.05, 36),
        }
    )
    master.to_csv(processed / "master_dataset.csv", index=False)

    # Write sample regime labels
    labels = ["Expansion", "Peak", "Contraction", "Recovery"]
    regimes = pd.DataFrame(
        {
            "Date": dates,
            "Regime": [labels[i % 4] for i in range(36)],
        }
    )
    regimes.to_csv(processed / "regime_labels.csv", index=False)

    # Write minimal risk metrics
    risk_rows = []
    sectors = ["Nifty_50_Return", "Nifty_Bank_Return", "Nifty_IT_Return"]
    for sector in sectors:
        for regime in ["All Regimes", "Expansion", "Peak", "Contraction", "Recovery"]:
            risk_rows.append(
                {
                    "Sector": sector,
                    "Regime": regime,
                    "Sharpe_Ratio": rng.normal(0.5, 0.3),
                    "Max_Drawdown": -rng.uniform(0.05, 0.3),
                    "VaR_95": -rng.uniform(0.02, 0.1),
                    "Months": 36,
                }
            )
    pd.DataFrame(risk_rows).to_csv(processed / "risk_metrics.csv", index=False)

    # Write empty backtest results
    (processed / "backtest_results.json").write_text("{}", encoding="utf-8")

    yield processed

    shutil.rmtree(tmpdir, ignore_errors=True)
