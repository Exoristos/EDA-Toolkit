"""Regression/modeling tests for eda_toolkit.analytics.modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eda_toolkit.analytics.modeling import fit_regression


# ═══════════════════════════════════════════════════════════════════════════════
# INLINE fixtures (MF-1: conftest.py'ye dokunma)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_linear_df(seed: int = 0, n: int = 200) -> pd.DataFrame:
    """Dogrusal iliski: y = 2*x1 - 3*x2 + 1 + noise."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-5, 5, n)
    x2 = rng.uniform(-5, 5, n)
    noise = rng.normal(0, 0.5, n)
    y = 2.0 * x1 - 3.0 * x2 + 1.0 + noise
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def _make_logistic_df(seed: int = 0, n: int = 200) -> pd.DataFrame:
    """Ikili hedef: skor = 1.5*x1 - 2*x2 + 0.5, sigmoid ile olasilik."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-3, 3, n)
    x2 = rng.uniform(-3, 3, n)
    score = 1.5 * x1 - 2.0 * x2 + 0.5 + rng.normal(0, 0.3, n)
    prob = 1.0 / (1.0 + np.exp(-score))
    y = (prob > 0.5).astype(int)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def _make_collinear_df(seed: int = 0, n: int = 100) -> pd.DataFrame:
    """x2 ≈ x1 ile yuksek VIF olusturur."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 10, n)
    x2 = x1 + rng.normal(0, 0.01, n)  # neredeyse esit
    y = 2.0 * x1 + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


# ═══════════════════════════════════════════════════════════════════════════════
# fit_regression tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFitLinearRegression:
    def test_recovers_coefficients_within_tolerance(self):
        df = _make_linear_df()
        result = fit_regression(df, target="y", features=["x1", "x2"], kind="linear")
        assert "error" not in result
        assert result["kind"] == "linear"
        coef = result["coefficients"]
        assert coef["x1"] == pytest.approx(2.0, abs=0.2)
        assert coef["x2"] == pytest.approx(-3.0, abs=0.2)

    def test_r2_above_threshold(self):
        df = _make_linear_df()
        result = fit_regression(df, target="y", features=["x1", "x2"], kind="linear")
        assert result["r2"] > 0.9

    def test_p_values_present(self):
        df = _make_linear_df()
        result = fit_regression(df, target="y", features=["x1", "x2"], kind="linear")
        assert "p_values" in result
        for feat in ["x1", "x2", "intercept"]:
            assert feat in result["p_values"]
            assert isinstance(result["p_values"][feat], float)

    def test_intercept_and_n_correct(self):
        df = _make_linear_df()
        result = fit_regression(df, target="y", features=["x1", "x2"], kind="linear")
        assert isinstance(result["intercept"], float)
        assert result["n"] == 200
        assert result["feature_names"] == ["x1", "x2"]

    def test_vif_present(self):
        df = _make_linear_df()  # x1, x2 bagimsiz → VIF ≈ 1
        result = fit_regression(df, target="y", features=["x1", "x2"], kind="linear")
        assert "vif" in result
        for feat in ["x1", "x2"]:
            assert 0.5 < result["vif"][feat] < 5.0
        assert result["vif_warning"] == []


class TestFitLogisticRegression:
    def test_accuracy_or_pseudo_r2_present(self):
        df = _make_logistic_df()
        result = fit_regression(df, target="y", features=["x1", "x2"], kind="logistic")
        assert "error" not in result
        assert result["kind"] == "logistic"
        has_accuracy = "accuracy" in result and isinstance(result["accuracy"], float)
        has_pseudo = "pseudo_r2" in result and isinstance(result["pseudo_r2"], float)
        assert has_accuracy or has_pseudo, "Loistik modelde accuracy veya pseudo_r2 bulunamadi."

    def test_accuracy_above_threshold(self):
        df = _make_logistic_df()
        result = fit_regression(df, target="y", features=["x1", "x2"], kind="logistic")
        if "accuracy" in result:
            assert result["accuracy"] > 0.7
        else:
            # pseudo_r2 varsa accuracy olmayabilir; yine de model calismis olmali
            assert result.get("pseudo_r2", 0) > 0.1

    def test_coefficients_present(self):
        df = _make_logistic_df()
        result = fit_regression(df, target="y", features=["x1", "x2"], kind="logistic")
        assert "coefficients" in result
        for feat in ["x1", "x2", "intercept"]:
            assert feat in result["coefficients"]

    def test_n_correct(self):
        df = _make_logistic_df()
        result = fit_regression(df, target="y", features=["x1", "x2"], kind="logistic")
        assert result["n"] == 200


class TestHighVIF:
    def test_collinear_features_trigger_vif_warning(self):
        df = _make_collinear_df()
        result = fit_regression(df, target="y", features=["x1", "x2"], kind="linear")
        assert "error" not in result
        assert result["vif_warning"] != [], f"VIF uyarisi bekleniyordu, vif={result['vif']}"


class TestTooFewRows:
    def test_insufficient_rows_returns_error(self):
        df = _make_linear_df(n=3)  # 3 satir, 2 ozellik → 2+1=3 gerekir, 3 yeterli
        # 4 ozellikli test
        df2 = pd.DataFrame({
            "y": [1.0, 2.0, 3.0],
            "x1": [0.1, 0.2, 0.3],
            "x2": [0.4, 0.5, 0.6],
            "x3": [0.7, 0.8, 0.9],
            "x4": [1.0, 1.1, 1.2],
        })
        result = fit_regression(df2, target="y", features=["x1", "x2", "x3", "x4"], kind="linear")
        assert "error" in result

    def test_too_few_rows_logistic(self):
        df = pd.DataFrame({
            "y": [0, 1, 0],
            "x1": [0.1, 0.2, 0.3],
            "x2": [0.4, 0.5, 0.6],
            "x3": [0.7, 0.8, 0.9],
        })
        result = fit_regression(df, target="y", features=["x1", "x2", "x3"], kind="logistic")
        assert "error" in result


class TestDefensiveBehavior:
    def test_unknown_kind_returns_error(self):
        df = _make_linear_df(n=10)
        result = fit_regression(df, target="y", features=["x1"], kind="ridge")
        assert "error" in result

    def test_missing_target_column(self):
        df = _make_linear_df(n=10)
        result = fit_regression(df, target="missing_col", features=["x1"], kind="linear")
        assert "error" in result

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["y", "x1"])
        result = fit_regression(df, target="y", features=["x1"], kind="linear")
        assert "error" in result


class TestOneHotEncoding:
    def test_categorical_feature_encoded(self):
        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame({
            "y": 2.0 * rng.uniform(1, 5, n) + rng.normal(0, 0.4, n),
            "x_num": rng.uniform(1, 5, n),
            "x_cat": rng.choice(["A", "B", "C"], n),
        })
        result = fit_regression(df, target="y", features=["x_num", "x_cat"], kind="linear")
        assert "error" not in result
        # One-hot sonrasi feature_names birgenislemis olmali
        fnames = result["feature_names"]
        assert len(fnames) >= 3  # x_num + 2 dummy (drop_first)
        assert "x_num" in fnames
        # katsayilar tum one-hot sutunlarini icermeli
        for fn in fnames:
            assert fn in result["coefficients"], f"{fn} katsayilarda yok"
        # VIF de tum feature'lar icin hesaplanmis olmali
        for fn in fnames:
            assert fn in result["vif"], f"{fn} VIF'te yok"
