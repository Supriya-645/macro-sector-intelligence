"""
Phase 1 — Data Pipeline for the Macro-Driven Sector Intelligence Platform.

Ingests 15+ years of equity, macro-economic, commodity, and volatility data
from three sources (yfinance, FRED, World Bank), cleans and resamples
everything to monthly frequency, and outputs a consolidated master dataset.

Usage
-----
    python src/data_pipeline.py

Outputs
-------
- data/raw/<source>_*.csv          — Raw downloaded data per source
- data/processed/master_dataset.csv — Consolidated monthly dataset
- data/processed/data_quality_report.txt — Summary of data quality
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (
    load_env,
    get_data_paths,
    get_project_root,
    SECTOR_INDICES,
    MARKET_TICKERS,
    FRED_SERIES,
    WORLD_BANK_INDICATORS,
    calculate_monthly_returns,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

START_DATE = "2009-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Data Ingestion Functions
# ---------------------------------------------------------------------------

def fetch_yfinance_data(tickers_dict, start, end):
    """Download daily price data from yfinance and resample to monthly.

    For each ticker, downloads the adjusted close price, resamples to
    month-end frequency using the last available observation, and returns
    a single merged DataFrame.

    Parameters
    ----------
    tickers_dict : dict
        Mapping of ``{column_name: yfinance_ticker}``.
    start : str
        Start date in ``YYYY-MM-DD`` format.
    end : str
        End date in ``YYYY-MM-DD`` format.

    Returns
    -------
    pandas.DataFrame
        Monthly price data indexed by month-end date.
    """
    all_data = {}
    failed = []

    for name, ticker in tickers_dict.items():
        print(f"  Downloading {name} ({ticker})...")
        try:
            data = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
            )
            if data.empty:
                print(f"     No data returned for {name}")
                failed.append(name)
                continue

            # Use 'Close' column; resample to month-end
            close = data["Close"].copy()
            # Handle MultiIndex columns from yfinance
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            monthly = close.resample("ME").last()
            all_data[name] = monthly
            print(f"    {len(monthly)} monthly observations")

        except Exception as e:
            print(f"    Failed: {e}")
            failed.append(name)

    if failed:
        print(f"\n   Failed tickers: {', '.join(failed)}")

    df = pd.DataFrame(all_data)
    df.index.name = "Date"
    return df


def fetch_fred_data(series_dict, api_key, start, end):
    """Download macroeconomic series from the FRED API.

    Handles mixed frequencies (daily, monthly, quarterly) by resampling
    everything to month-end. Daily series take the monthly mean;
    quarterly/annual series are forward-filled.

    Parameters
    ----------
    series_dict : dict
        Mapping of ``{column_name: FRED_series_id}``.
    api_key : str
        FRED API key.
    start : str
        Start date in ``YYYY-MM-DD`` format.
    end : str
        End date in ``YYYY-MM-DD`` format.

    Returns
    -------
    pandas.DataFrame
        Monthly macro data indexed by month-end date.
    """
    if not api_key or api_key == "your_fred_api_key_here":
        print("   FRED API key not configured — skipping FRED data.")
        return pd.DataFrame()

    try:
        from fredapi import Fred
    except ImportError:
        print("  fredapi not installed. Run: pip install fredapi")
        return pd.DataFrame()

    fred = Fred(api_key=api_key)
    all_data = {}
    failed = []

    for name, series_id in series_dict.items():
        print(f"  Downloading {name} ({series_id}) from FRED...")
        try:
            series = fred.get_series(
                series_id,
                observation_start=start,
                observation_end=end,
            )
            if series.empty:
                print(f"     No data for {name}")
                failed.append(name)
                continue

            # Detect frequency and resample appropriately
            inferred_freq = pd.infer_freq(series.index)

            if inferred_freq and inferred_freq.startswith(("B", "D")):
                # Daily data → take monthly mean
                monthly = series.resample("ME").mean()
            elif inferred_freq and inferred_freq.startswith("Q"):
                # Quarterly → forward-fill to monthly
                monthly = series.resample("ME").ffill()
            elif inferred_freq and inferred_freq.startswith("A"):
                # Annual → forward-fill to monthly
                monthly = series.resample("ME").ffill()
            else:
                # Already monthly or unknown — resample to be safe
                monthly = series.resample("ME").last()

            all_data[name] = monthly
            print(f"    {len(monthly)} monthly observations")

        except Exception as e:
            print(f"    Failed: {e}")
            failed.append(name)

    if failed:
        print(f"\n   Failed FRED series: {', '.join(failed)}")

    df = pd.DataFrame(all_data)
    df.index.name = "Date"
    return df


def fetch_world_bank_data(indicators_dict, country_code="IND",
                          start_year=2009):
    """Download annual indicators from the World Bank and resample monthly.

    Uses the ``wbgapi`` library. Annual data is forward-filled to create
    a monthly series, meaning each month within a year carries the same
    annual value until the next year's data is available.

    Parameters
    ----------
    indicators_dict : dict
        Mapping of ``{column_name: WB_indicator_code}``.
    country_code : str, optional
        ISO 3-letter country code (default ``'IND'`` for India).
    start_year : int, optional
        Earliest year to fetch (default 2009).

    Returns
    -------
    pandas.DataFrame
        Monthly data (forward-filled from annual) indexed by month-end date.
    """
    try:
        import wbgapi as wb
    except ImportError:
        print("  wbgapi not installed. Run: pip install wbgapi")
        return pd.DataFrame()

    all_data = {}
    current_year = datetime.now().year

    for name, indicator in indicators_dict.items():
        print(f"  Downloading {name} ({indicator}) from World Bank...")
        try:
            # wbgapi returns a DataFrame with years as columns
            result = wb.data.DataFrame(
                indicator,
                economy=country_code,
                time=range(start_year, current_year + 1),
                labels=False,
                columns="time",
            )

            if result.empty:
                print(f"     No data for {name}")
                continue

            # Transpose: years become rows
            series = result.iloc[0]  # Single country, single indicator
            # Convert year strings like "YR2009" to datetime
            dates = []
            values = []
            for col_name, val in series.items():
                year_str = str(col_name).replace("YR", "")
                try:
                    year = int(year_str)
                    dates.append(pd.Timestamp(f"{year}-12-31"))
                    values.append(val)
                except ValueError:
                    continue

            annual_series = pd.Series(values, index=pd.DatetimeIndex(dates),
                                       name=name)
            annual_series = annual_series.sort_index()

            # Resample to monthly with forward fill
            monthly = annual_series.resample("ME").ffill()
            all_data[name] = monthly
            print(f"    {len(monthly)} monthly observations "
                  f"(from {len(annual_series)} annual)")

        except Exception as e:
            print(f"    Failed: {e}")

    df = pd.DataFrame(all_data)
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# Data Cleaning & Merging
# ---------------------------------------------------------------------------

def clean_and_merge(equity_df, fred_df, wb_df):
    """Merge all data sources into a single monthly DataFrame.

    Aligns all DataFrames on a common monthly date index, applies
    forward-fill for minor gaps, and computes monthly percentage
    returns for all equity/sector indices.

    Parameters
    ----------
    equity_df : pandas.DataFrame
        Monthly equity and market data from yfinance.
    fred_df : pandas.DataFrame
        Monthly macro data from FRED.
    wb_df : pandas.DataFrame
        Monthly (forward-filled annual) data from the World Bank.

    Returns
    -------
    pandas.DataFrame
        Consolidated master dataset with levels and returns.
    """
    # Merge on date index
    dfs = [df for df in [equity_df, fred_df, wb_df] if not df.empty]
    if not dfs:
        raise ValueError("No data frames to merge!")

    master = dfs[0]
    for df in dfs[1:]:
        master = master.join(df, how="outer")

    # Sort by date
    master.sort_index(inplace=True)

    # Forward-fill gaps (max 3 months to avoid hallucinating long gaps)
    master = master.ffill(limit=3)

    # Compute monthly returns for sector/equity indices
    sector_cols = [col for col in SECTOR_INDICES.keys() if col in master.columns]
    for col in sector_cols:
        master[f"{col}_Return"] = calculate_monthly_returns(master[col])

    # Also compute returns for Nifty 50 specifically
    if "Nifty_50" in master.columns:
        master["Nifty_50_Return"] = calculate_monthly_returns(master["Nifty_50"])

    # Drop rows where ALL values are NaN (months before any data starts)
    master.dropna(how="all", inplace=True)

    return master


# ---------------------------------------------------------------------------
# Data Quality Report
# ---------------------------------------------------------------------------

def generate_quality_report(df, output_path):
    """Generate a text-based data quality report.

    Documents null counts, date coverage, basic statistics, and
    data completeness per column.

    Parameters
    ----------
    df : pandas.DataFrame
        The master dataset to report on.
    output_path : pathlib.Path
        Path to write the report file.
    """
    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  MACRO-DRIVEN SECTOR INTELLIGENCE PLATFORM\n")
        f.write("  Data Quality Report\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
        f.write(f"Date Range: {df.index.min()} → {df.index.max()}\n")
        f.write(f"Frequency: Monthly (ME)\n\n")

        # Null counts
        f.write("-" * 70 + "\n")
        f.write("  NULL COUNTS PER COLUMN\n")
        f.write("-" * 70 + "\n")
        nulls = df.isnull().sum()
        total = len(df)
        for col in df.columns:
            n = nulls[col]
            pct = (n / total) * 100
            status = "OK" if pct < 5 else ("WARN" if pct < 20 else "ERR")
            f.write(f"  {status} {col:<30s}  {n:>4d} nulls  ({pct:5.1f}%)\n")

        f.write(f"\n  Total cells: {df.size}\n")
        f.write(f"  Total nulls: {df.isnull().sum().sum()}\n")
        f.write(f"  Overall completeness: "
                f"{(1 - df.isnull().sum().sum() / df.size) * 100:.1f}%\n\n")

        # Descriptive statistics
        f.write("-" * 70 + "\n")
        f.write("  DESCRIPTIVE STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(df.describe().to_string())
        f.write("\n\n")

        # Column data types
        f.write("-" * 70 + "\n")
        f.write("  COLUMN DATA TYPES\n")
        f.write("-" * 70 + "\n")
        for col in df.columns:
            f.write(f"  {col:<30s}  {str(df[col].dtype)}\n")

    print(f"  Quality report saved: {output_path}")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def run_pipeline():
    """Execute the full data ingestion pipeline.

    Orchestrates downloading from all three sources, merging,
    cleaning, and saving the master dataset along with raw
    components and a quality report.
    """
    print("=" * 60)
    print("  PHASE 1 — DATA PIPELINE")
    print("=" * 60)

    # Load environment
    env = load_env()
    paths = get_data_paths()

    # Ensure output directories exist
    paths["raw"].mkdir(parents=True, exist_ok=True)
    paths["processed"].mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Fetch yfinance data (equity indices + market indicators)
    # ------------------------------------------------------------------
    print("\nStep 1/4: Fetching yfinance data...")
    all_yf_tickers = {**SECTOR_INDICES, **MARKET_TICKERS}
    equity_df = fetch_yfinance_data(all_yf_tickers, START_DATE, END_DATE)

    if not equity_df.empty:
        equity_df.to_csv(paths["raw"] / "yfinance_data.csv")
        print(f"  Saved raw yfinance data: {equity_df.shape}")

    # ------------------------------------------------------------------
    # Step 2: Fetch FRED data (US macro + commodities)
    # ------------------------------------------------------------------
    print("\n Step 2/4: Fetching FRED data...")
    fred_df = fetch_fred_data(FRED_SERIES, env["FRED_API_KEY"],
                               START_DATE, END_DATE)

    if not fred_df.empty:
        fred_df.to_csv(paths["raw"] / "fred_data.csv")
        print(f"  Saved raw FRED data: {fred_df.shape}")

    # ------------------------------------------------------------------
    # Step 3: Fetch World Bank data (India macro)
    # ------------------------------------------------------------------
    print("\nStep 3/4: Fetching World Bank data...")
    wb_df = fetch_world_bank_data(WORLD_BANK_INDICATORS)

    if not wb_df.empty:
        wb_df.to_csv(paths["raw"] / "worldbank_data.csv")
        print(f"  Saved raw World Bank data: {wb_df.shape}")

    # ------------------------------------------------------------------
    # Step 4: Clean, merge, and save
    # ------------------------------------------------------------------
    print("\nStep 4/4: Cleaning and merging...")
    master = clean_and_merge(equity_df, fred_df, wb_df)

    # Save master dataset
    master_path = paths["processed"] / "master_dataset.csv"
    master.to_csv(master_path)
    print(f"\n  Master dataset saved: {master_path}")
    print(f"     Shape: {master.shape[0]} rows × {master.shape[1]} columns")
    print(f"     Date range: {master.index.min().date()} → "
          f"{master.index.max().date()}")

    # Generate quality report
    report_path = paths["processed"] / "data_quality_report.txt"
    generate_quality_report(master, report_path)

    # Summary
    print("\n" + "=" * 60)
    print("  PHASE 1 COMPLETE")
    print("=" * 60)
    print(f"  Master dataset: {master_path}")
    print(f"  Quality report: {report_path}")
    print(f"  Columns: {list(master.columns)}")

    return master


if __name__ == "__main__":
    run_pipeline()
