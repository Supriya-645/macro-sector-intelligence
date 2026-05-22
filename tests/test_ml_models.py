"""Tests for the optuna_tuning module — parameter structure and helpers."""

import json
from pathlib import Path

import pytest


def test_optuna_module_importable() -> None:
    """Optuna tuning module should import without error.

    Verifies the module structure and public API are correct.
    """
    try:
        import src.optuna_tuning as opt  # noqa: F401
        assert hasattr(opt, "run_tuning")
        assert hasattr(opt, "load_best_params")
    except ImportError:
        pytest.skip("optuna or xgboost not installed.")


def test_load_best_params_returns_dict(tmp_data_dir) -> None:
    """load_best_params should return a dict (empty if no file exists).

    Args:
        tmp_data_dir: Session-scoped temp data directory fixture.
    """
    try:
        from unittest.mock import patch
        import src.optuna_tuning as opt

        # Patch get_data_paths to use tmp_data_dir
        mock_paths = {"processed": tmp_data_dir, "models": tmp_data_dir / "models"}
        with patch("src.optuna_tuning.get_data_paths", return_value=mock_paths):
            result = opt.load_best_params("Nifty_50_Return")
        assert isinstance(result, dict)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("Dependencies not available.")


def test_load_best_params_reads_json(tmp_data_dir) -> None:
    """load_best_params should correctly parse an existing optuna_params.json.

    Args:
        tmp_data_dir: Session-scoped temp data directory fixture.
    """
    try:
        from unittest.mock import patch
        import src.optuna_tuning as opt

        models_dir = tmp_data_dir / "models"
        models_dir.mkdir(exist_ok=True)
        sample_params = {
            "Nifty_50_Return": {
                "params": {"learning_rate": 0.05, "max_depth": 4},
                "best_f1": 0.62,
            }
        }
        (models_dir / "optuna_params.json").write_text(
            json.dumps(sample_params), encoding="utf-8"
        )

        mock_paths = {"processed": tmp_data_dir, "models": models_dir}
        with patch("src.optuna_tuning.get_data_paths", return_value=mock_paths):
            result = opt.load_best_params("Nifty_50_Return")

        assert result["learning_rate"] == 0.05
        assert result["max_depth"] == 4
    except (ImportError, ModuleNotFoundError):
        pytest.skip("Dependencies not available.")


def test_ml_models_loads_tuned_params(monkeypatch) -> None:
    """ml_models should expose tuned XGBoost params to training helpers."""
    try:
        import src.ml_models as ml
        import src.optuna_tuning  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        pytest.skip("XGBoost/Optuna dependencies not available.")

    expected = {"learning_rate": 0.05, "max_depth": 4}

    def fake_load_best_params(sector_name):
        assert sector_name == "Nifty_50_Return"
        return expected

    monkeypatch.setattr("src.optuna_tuning.load_best_params", fake_load_best_params)
    assert ml._load_tuned_xgb_params("Nifty_50_Return") == expected


def test_xgb_classifier_merges_default_and_tuned_params() -> None:
    """Tuned values should override defaults while preserving required defaults."""
    try:
        import src.ml_models as ml
    except (ImportError, ModuleNotFoundError):
        pytest.skip("XGBoost dependencies not available.")

    model = ml._build_xgb_classifier({"learning_rate": 0.03, "max_depth": 5})

    assert model.learning_rate == 0.03
    assert model.max_depth == 5
    assert model.random_state == 42
