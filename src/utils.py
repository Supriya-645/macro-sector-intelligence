"""
Shared utility functions for the Macro-Driven Sector Intelligence Platform.

Provides common helpers for path resolution, data loading, chart styling,
and return calculations used across all phases of the pipeline.
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "macro_matplotlib"))

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Path & Environment Helpers
# ---------------------------------------------------------------------------

def get_project_root():
    """Return the absolute path to the project root directory.

    Walks up from this file's location to find the directory containing
    'requirements.txt', which serves as the project root marker.

    Returns
    -------
    pathlib.Path
        Absolute path to the project root.

    Raises
    ------
    FileNotFoundError
        If the project root cannot be located.
    """
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "requirements.txt").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate project root (requirements.txt)")


def load_env():
    """Load environment variables from the project's ``.env`` file.

    Returns
    -------
    dict
        Dictionary containing loaded environment variables of interest,
        currently ``{'FRED_API_KEY': ...}``.

    Raises
    ------
    EnvironmentError
        If a required API key is missing or set to the default placeholder.
    """
    root = get_project_root()
    dotenv_path = root / ".env"
    load_dotenv(dotenv_path)

    fred_key = os.getenv("FRED_API_KEY", "")
    if not fred_key or fred_key == "your_fred_api_key_here":
        print(
            " WARNING: FRED_API_KEY not set or still placeholder. "
            "FRED data will be skipped. Set it in .env to enable."
        )

    return {"FRED_API_KEY": fred_key}


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def get_data_paths():
    """Return a dict of commonly used data directory paths.

    Returns
    -------
    dict
        Keys: 'raw', 'processed', 'charts', 'models'.
    """
    root = get_project_root()
    return {
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "charts": root / "data" / "processed" / "charts",
        "models": root / "data" / "processed" / "models",
    }


def load_master_dataset():
    """Load the consolidated master dataset from CSV.

    Parses the ``Date`` column as the index and sorts chronologically.

    Returns
    -------
    pandas.DataFrame
        The master dataset with a ``DatetimeIndex``.
    """
    paths = get_data_paths()
    csv_path = paths["processed"] / "master_dataset.csv"
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    df.sort_index(inplace=True)
    return df


def load_regime_labels():
    """Load regime labels produced by Phase 4.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ``Date`` index and a ``Regime`` column.
    """
    paths = get_data_paths()
    csv_path = paths["processed"] / "regime_labels.csv"
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    return df


# ---------------------------------------------------------------------------
# Return Calculations
# ---------------------------------------------------------------------------

def calculate_monthly_returns(series, periods=1):
    """Calculate percentage returns over *periods* months.

    Parameters
    ----------
    series : pandas.Series
        Time series of price levels.
    periods : int, optional
        Number of periods for return calculation (default 1).

    Returns
    -------
    pandas.Series
        Percentage returns (e.g., 0.05 for 5 %).
    """
    return series.pct_change(periods=periods)


def calculate_log_returns(series, periods=1):
    """Calculate logarithmic returns over *periods* months.

    Parameters
    ----------
    series : pandas.Series
        Time series of price levels.
    periods : int, optional
        Number of periods for return calculation (default 1).

    Returns
    -------
    pandas.Series
        Log returns.
    """
    return np.log(series / series.shift(periods))


# ---------------------------------------------------------------------------
# Plotting Style
# ---------------------------------------------------------------------------

# Premium colour palette inspired by dark-mode financial terminals
PALETTE = {
    "bg_dark": "#0D1117",
    "bg_card": "#161B22",
    "text": "#E6EDF3",
    "text_muted": "#8B949E",
    "accent_blue": "#58A6FF",
    "accent_green": "#3FB950",
    "accent_red": "#F85149",
    "accent_orange": "#D29922",
    "accent_purple": "#BC8CFF",
    "grid": "#21262D",
}

REGIME_COLORS = {
    "Expansion": "#3FB950",
    "Peak": "#D29922",
    "Contraction": "#F85149",
    "Recovery": "#58A6FF",
}

SECTOR_COLORS = [
    "#58A6FF", "#3FB950", "#F85149", "#D29922", "#BC8CFF",
    "#F778BA", "#79C0FF", "#56D364", "#FFA657", "#FF7B72",
    "#D2A8FF", "#A5D6FF",
]


def setup_plotting_style():
    """Configure matplotlib and seaborn with a premium dark theme.

    Sets global rcParams for a consistent, publication-quality look
    across all generated charts.
    """
    plt.style.use("dark_background")
    mpl.rcParams.update({
        "figure.facecolor": PALETTE["bg_dark"],
        "axes.facecolor": PALETTE["bg_card"],
        "axes.edgecolor": PALETTE["grid"],
        "axes.labelcolor": PALETTE["text"],
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.alpha": 0.4,
        "text.color": PALETTE["text"],
        "xtick.color": PALETTE["text_muted"],
        "ytick.color": PALETTE["text_muted"],
        "legend.facecolor": PALETTE["bg_card"],
        "legend.edgecolor": PALETTE["grid"],
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": PALETTE["bg_dark"],
    })
    sns.set_palette(SECTOR_COLORS)


def save_chart(fig, name, subfolder=""):
    """Save a matplotlib figure to the charts directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    name : str
        Filename (without extension) for the saved PNG.
    subfolder : str, optional
        Subdirectory under ``charts/`` (e.g. ``'eda'``, ``'regime'``).
    """
    paths = get_data_paths()
    out_dir = paths["charts"] / subfolder if subfolder else paths["charts"]
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / f"{name}.png"
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved chart: {filepath.relative_to(get_project_root())}")


# ---------------------------------------------------------------------------
# Sector & Indicator Definitions
# ---------------------------------------------------------------------------

# Canonical lists used across all phases for consistency.

SECTOR_INDICES = {
    "Nifty_50": "^NSEI",
    "Nifty_Bank": "^NSEBANK",
    "Nifty_IT": "^CNXIT",
    "Nifty_Pharma": "^CNXPHARMA",
    "Nifty_Auto": "^CNXAUTO",
    "Nifty_FMCG": "^CNXFMCG",
    "Nifty_Metal": "^CNXMETAL",
    "Nifty_Realty": "^CNXREALTY",
    "Nifty_Energy": "^CNXENERGY",
    "Nifty_Infra": "^CNXINFRA",
    "Nifty_PSE": "^CNXPSE",
    "Nifty_Media": "^CNXMEDIA",
}

MARKET_TICKERS = {
    "India_VIX": "^INDIAVIX",
    "USD_INR": "USDINR=X",
    "DXY": "DX-Y.NYB",
    "Global_VIX": "^VIX",
}

FRED_SERIES = {
    "US_Fed_Funds_Rate": "FEDFUNDS",
    "US_CPI": "CPIAUCSL",
    "US_10Y_Yield": "DGS10",
    "US_GDP": "GDP",
    "Brent_Crude": "DCOILBRENTEU",
    "Gold_Price": "GOLDAMGBD228NLBM",
}

WORLD_BANK_INDICATORS = {
    "India_GDP": "NY.GDP.MKTP.CD",
    "India_CPI": "FP.CPI.TOTL.ZG",
}

# Crisis / landmark event windows for EDA
EVENT_WINDOWS = {
    "2008 GFC": ("2008-09-01", "2009-03-31"),
    "2013 Taper Tantrum": ("2013-05-01", "2013-09-30"),
    "2020 COVID Crash": ("2020-02-01", "2020-05-31"),
    "2022 Rate Hikes": ("2022-01-01", "2022-10-31"),
}

# Columns that represent sector return series (computed in pipeline)
SECTOR_RETURN_COLS = [f"{name}_Return" for name in SECTOR_INDICES.keys()]

# Macro indicator columns (levels, not returns)
MACRO_COLS = (
    list(FRED_SERIES.keys())
    + list(WORLD_BANK_INDICATORS.keys())
    + list(MARKET_TICKERS.keys())
)
