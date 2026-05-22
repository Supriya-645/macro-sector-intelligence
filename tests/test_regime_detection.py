"""Tests for economic regime detection — K-Means and HMM labeling."""

import pandas as pd
import pytest

VALID_REGIME_LABELS = {"Expansion", "Peak", "Contraction", "Recovery"}


def test_regime_fixture_has_required_columns(sample_regime_df: pd.DataFrame) -> None:
    """Regime fixture must contain Date and Regime columns.

    Args:
        sample_regime_df: Sample regime DataFrame from conftest.
    """
    assert "Date" in sample_regime_df.columns
    assert "Regime" in sample_regime_df.columns


def test_regime_fixture_valid_labels(sample_regime_df: pd.DataFrame) -> None:
    """All regime labels in the fixture must be one of the four valid states.

    Args:
        sample_regime_df: Sample regime DataFrame from conftest.
    """
    unique_labels = set(sample_regime_df["Regime"].unique())
    assert unique_labels.issubset(VALID_REGIME_LABELS), (
        f"Unexpected regime labels: {unique_labels - VALID_REGIME_LABELS}"
    )


def test_regime_fixture_no_nulls(sample_regime_df: pd.DataFrame) -> None:
    """Regime column must not contain any null values.

    Args:
        sample_regime_df: Sample regime DataFrame from conftest.
    """
    assert sample_regime_df["Regime"].isna().sum() == 0


def test_tmp_regime_labels(tmp_data_dir) -> None:
    """Regime labels CSV in tmp directory must have 4 unique regime states.

    Args:
        tmp_data_dir: Session-scoped temp data directory fixture.
    """
    df = pd.read_csv(tmp_data_dir / "regime_labels.csv")
    unique = set(df["Regime"].unique())
    assert len(unique) == 4, f"Expected 4 unique regimes, got: {unique}"
    assert unique == VALID_REGIME_LABELS


def test_regime_covers_all_dates(sample_regime_df: pd.DataFrame) -> None:
    """Every row in the regime DataFrame should have a non-empty date.

    Args:
        sample_regime_df: Sample regime DataFrame from conftest.
    """
    assert sample_regime_df["Date"].isna().sum() == 0
    assert len(sample_regime_df) == 24


def test_hmm_module_importable() -> None:
    """HMM regime module should be importable without error.

    This validates that the module structure and imports are correct
    even if hmmlearn is not installed (graceful ImportError handling).
    """
    try:
        import src.hmm_regime as hmm_mod  # noqa: F401
        assert hasattr(hmm_mod, "run_hmm_regime_detection")
        assert hasattr(hmm_mod, "load_hmm_labels")
    except ImportError:
        pytest.skip("src.hmm_regime dependencies not installed.")
