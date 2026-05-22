"""Tests for the data pipeline — master dataset integrity checks."""

import pandas as pd
import pytest


EXPECTED_MACRO_COLS = [
    "US_Fed_Funds_Rate",
    "US_CPI",
    "US_10Y_Yield",
    "Brent_Crude",
    "Gold_Price",
    "India_VIX",
    "USD_INR",
    "DXY",
]

EXPECTED_RETURN_COLS = [
    "Nifty_50_Return",
    "Nifty_Bank_Return",
    "Nifty_IT_Return",
]


def test_sample_master_df_shape(sample_master_df: pd.DataFrame) -> None:
    """Master dataset fixture should have 24 rows and required columns.

    Args:
        sample_master_df: Sample DataFrame fixture from conftest.
    """
    assert len(sample_master_df) == 24
    assert "Date" in sample_master_df.columns


def test_master_numeric_columns(sample_master_df: pd.DataFrame) -> None:
    """All macro and return columns should be numeric (float/int).

    Args:
        sample_master_df: Sample DataFrame fixture from conftest.
    """
    for col in EXPECTED_MACRO_COLS + EXPECTED_RETURN_COLS:
        if col in sample_master_df.columns:
            assert pd.api.types.is_numeric_dtype(sample_master_df[col]), (
                f"Column '{col}' is not numeric."
            )


def test_no_duplicate_dates(sample_master_df: pd.DataFrame) -> None:
    """The Date column must not contain any duplicate values.

    Args:
        sample_master_df: Sample DataFrame fixture from conftest.
    """
    assert sample_master_df["Date"].duplicated().sum() == 0


def test_dates_monotonic(sample_master_df: pd.DataFrame) -> None:
    """Dates in the master dataset must be in ascending order.

    Args:
        sample_master_df: Sample DataFrame fixture from conftest.
    """
    dates = pd.to_datetime(sample_master_df["Date"])
    assert dates.is_monotonic_increasing, "Dates are not in ascending order."


def test_nifty_50_positive(sample_master_df: pd.DataFrame) -> None:
    """Nifty 50 index values should always be positive.

    Args:
        sample_master_df: Sample DataFrame fixture from conftest.
    """
    if "Nifty_50" in sample_master_df.columns:
        assert (sample_master_df["Nifty_50"].dropna() > 0).all(), (
            "Nifty 50 contains non-positive values."
        )


def test_tmp_data_dir_has_master_csv(tmp_data_dir) -> None:
    """Temporary data directory should contain master_dataset.csv.

    Args:
        tmp_data_dir: Session-scoped fixture providing a temp processed/ dir.
    """
    assert (tmp_data_dir / "master_dataset.csv").exists()


def test_tmp_master_csv_readable(tmp_data_dir) -> None:
    """master_dataset.csv in tmp_data_dir should be loadable as a DataFrame.

    Args:
        tmp_data_dir: Session-scoped fixture providing a temp processed/ dir.
    """
    df = pd.read_csv(tmp_data_dir / "master_dataset.csv")
    assert not df.empty
    assert "Date" in df.columns


def test_tmp_regime_csv_readable(tmp_data_dir) -> None:
    """regime_labels.csv in tmp_data_dir should be loadable with Regime column.

    Args:
        tmp_data_dir: Session-scoped fixture providing a temp processed/ dir.
    """
    df = pd.read_csv(tmp_data_dir / "regime_labels.csv")
    assert "Regime" in df.columns
    assert not df.empty
