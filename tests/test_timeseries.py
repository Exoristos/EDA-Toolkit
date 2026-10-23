"""Zaman serisi ayrıştırma ve tahmin testleri (inline fixture)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from eda_toolkit.analytics.timeseries import decompose, forecast


# ── Inline fixture ────────────────────────────────────────────────────────────


def _make_ts(n: int = 60, period: int = 12) -> pd.DataFrame:
    """Deterministik sentetik seri: DatetimeIndex + trend + sinüs mevsimselligi + gürültü.

    Sabit ``numpy.random.seed(0)`` ile deterministik.
    """
    np.random.seed(0)
    dates = pd.date_range(start="2020-01-01", periods=n, freq="MS")
    trend = np.arange(n, dtype=float) * 0.5
    seasonal = 5.0 * np.sin(2.0 * np.pi * np.arange(n) / period)
    noise = np.random.normal(0, 0.5, size=n)
    values = trend + seasonal + noise
    return pd.DataFrame({"deger": values}, index=dates)


# ═══════════════════════════════════════════════════════════════════════════════
# decompose
# ═══════════════════════════════════════════════════════════════════════════════


class TestDecompose:
    def test_returns_expected_keys(self):
        df = _make_ts(60, 12)
        result = decompose(df, "deger", period=12)
        expected_keys = {"trend", "seasonal", "residual", "period", "n"}
        assert set(result.keys()) == expected_keys

    def test_lengths_match_input(self):
        df = _make_ts(60, 12)
        result = decompose(df, "deger", period=12)
        n = result["n"]
        assert len(result["trend"]) == n
        assert len(result["seasonal"]) == n
        assert len(result["residual"]) == n

    def test_nan_replaced_with_none(self):
        """Başlangıçta NaN olan trend değerleri None'a çevrilmiş olmalı."""
        df = _make_ts(60, 12)
        result = decompose(df, "deger", period=12)
        # seasonal_decompose additive extrapolate_trend=0 → başta 6 NaN (period/2)
        # ilk period/2 kadar None beklenir
        none_count = sum(1 for v in result["trend"] if v is None)
        assert none_count > 0

    def test_missing_column(self):
        df = _make_ts(30, 7)
        result = decompose(df, "yok_sutun", period=7)
        assert "error" in result
        assert "yok_sutun" in result["error"]

    def test_short_series(self):
        df = _make_ts(10, 12)  # 10 < 2*12=24
        result = decompose(df, "deger", period=12)
        assert "error" in result
        assert result["period"] == 12

    def test_non_numeric_column(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"] * 10})
        result = decompose(df, "cat", period=3)
        assert "error" in result

    def test_auto_period_datetimeindex_monthly(self):
        df = _make_ts(48, 12)
        # DatetimeIndex + frekans MS → 12 olarak tahmin edilmeli
        result = decompose(df, "deger")
        assert "error" not in result
        assert result["period"] == 12

    def test_no_period_no_datetimeindex(self):
        df = pd.DataFrame({"deger": np.arange(30, dtype=float)})
        result = decompose(df, "deger")
        assert "error" in result
        assert "period" in result["error"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# forecast
# ═══════════════════════════════════════════════════════════════════════════════


class TestForecast:
    def test_arima_returns_expected_keys(self):
        df = _make_ts(60, 12)
        result = forecast(df, "deger", horizon=6, method="arima")
        expected = {"point", "ci_lower", "ci_upper", "method", "horizon", "order", "warning"}
        assert set(result.keys()) == expected
        assert result["method"] == "arima"
        assert result["horizon"] == 6
        # _make_ts DatetimeIndex kullandigi icin uyari yok (None).
        assert result["warning"] is None

    def test_warns_when_no_datetime_index(self):
        """MF-1 (ora-9): DatetimeIndex yoksa anlamli bir uyari donmeli."""
        np.random.seed(0)
        df = pd.DataFrame({"deger": np.arange(60, dtype=float) + np.random.randn(60)})
        result = forecast(df, "deger", horizon=4, method="arima")
        assert "error" not in result
        assert result["warning"] is not None
        assert "DatetimeIndex" in result["warning"]

    def test_arima_lengths_match_horizon(self):
        df = _make_ts(60, 12)
        result = forecast(df, "deger", horizon=6, method="arima")
        assert len(result["point"]) == 6
        assert len(result["ci_lower"]) == 6
        assert len(result["ci_upper"]) == 6

    def test_arima_with_period_warns_and_upgrades(self):
        df = _make_ts(60, 12)
        with pytest.warns(UserWarning, match="SARIMAX kullaniliyor"):
            result = forecast(df, "deger", horizon=6, method="arima", period=12)
        assert result["method"] == "sarimax"
        assert "x" in result["order"]  # mevsimsel order içerir

    def test_explicit_sarimax(self):
        df = _make_ts(60, 12)
        result = forecast(df, "deger", horizon=4, method="sarimax", period=12)
        assert result["method"] == "sarimax"
        assert len(result["point"]) == 4

    def test_short_series_error(self):
        df = _make_ts(8, 12)  # 8 < 10
        result = forecast(df, "deger", horizon=4, method="arima")
        assert "error" in result
        assert kisa_seri_kontrol(result["error"])

    def test_missing_column(self):
        df = _make_ts(30, 7)
        result = forecast(df, "yok_sutun", horizon=4, method="arima")
        assert "error" in result
        assert "yok_sutun" in result["error"]

    def test_unknown_method_raises(self):
        df = _make_ts(30, 7)
        with pytest.raises(ValueError, match="Bilinmeyen tahmin yontemi"):
            forecast(df, "deger", horizon=4, method="auto")

    def test_non_numeric_column(self):
        df = pd.DataFrame({"cat": ["x", "y"] * 10})
        result = forecast(df, "cat", horizon=3, method="arima")
        assert "error" in result

    def test_defensive_fit_error(self):
        """Convergence zor bir seride hata yükseltilmez, dict ile döner."""
        # Tümü aynı değer → varyans yok, fit patlayabilir
        df = pd.DataFrame(
            {"y": [1.0] * 20},
            index=pd.date_range("2020-01-01", periods=20, freq="MS"),
        )
        result = forecast(df, "y", horizon=2, method="arima")
        # Ya başarılı ya da error olabilir — ama raise edilmemeli
        assert isinstance(result, dict)
        if "error" in result:
            assert isinstance(result["error"], str)


def kisa_seri_kontrol(msg: str) -> bool:
    """Türkçe 'kisa' veya 'yetersiz' geçiyor mu?"""
    return "kisa" in msg.lower() or "yetersiz" in msg.lower()
