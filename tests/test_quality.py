"""Tests for eda_toolkit.analytics.quality — data_quality_score."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eda_toolkit.analytics.quality import data_quality_score


# ── Inline fixtures (MF-1: no conftest.py edits) ──────────────────────────────

def _clean_df() -> pd.DataFrame:
    """Tamamen temiz, eksiksiz bir DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        "sayisal_a": np.random.normal(50, 10, 100),
        "sayisal_b": np.random.normal(30, 5, 100),
        "kategori": np.random.choice(["A", "B", "C"], 100),
    })


def _missing_df() -> pd.DataFrame:
    """%50 eksik veri iceren DataFrame."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "sayisal_a": np.random.normal(50, 10, n),
        "sayisal_b": np.random.normal(30, 5, n),
        "kategori": np.random.choice(["A", "B", "C"], n),
    })
    # %50 NaN enjekte et
    mask = np.random.random((n, 3)) < 0.5
    df[mask] = np.nan
    return df


def _duplicate_df() -> pd.DataFrame:
    """Tekrarli satir iceren DataFrame."""
    df = pd.DataFrame({
        "a": [1, 2, 3, 1, 2],
        "b": ["x", "y", "z", "x", "y"],
    })
    return df


def _outlier_df() -> pd.DataFrame:
    """Asiri deger iceren DataFrame."""
    return pd.DataFrame({
        "normal": [10, 12, 11, 13, 10, 12, 11, 10, 13, 12],
        "outlier_var": [10, 12, 11, 13, 10, 100, 11, 10, 13, -50],
    })


def _constant_col_df() -> pd.DataFrame:
    """Sabit sutun iceren DataFrame."""
    return pd.DataFrame({
        "degisen": [1, 2, 3, 4],
        "sabit_1": [5, 5, 5, 5],
        "sabit_null": [np.nan, np.nan, np.nan, np.nan],
    })


def _empty_df() -> pd.DataFrame:
    """Bos DataFrame."""
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataQualityScore:
    """data_quality_score fonksiyonu testleri."""

    def test_clean_df_high_score(self):
        """Temiz bir DataFrame yuksek kalite skoru almali."""
        result = data_quality_score(_clean_df())
        assert "score" in result
        assert isinstance(result["score"], float)
        # Temiz veri: missing=0, dup=0, dusuk outlier, sabit sutun yok
        assert result["score"] >= 90.0, f"Expected >=90, got {result['score']}"

    def test_return_structure(self):
        """Donus sozlugu tum anahtarlari icermeli."""
        result = data_quality_score(_clean_df())
        expected_keys = {
            "score", "missing_rate", "duplicate_rate",
            "outlier_rate", "constant_columns", "components",
        }
        assert expected_keys <= set(result.keys()), f"Missing keys: {expected_keys - set(result.keys())}"
        assert 0.0 <= result["score"] <= 100.0

    def test_missing_df_lower_score(self):
        """Eksik verili DataFrame daha dusuk skor almali."""
        clean = data_quality_score(_clean_df())
        missing = data_quality_score(_missing_df())
        assert missing["score"] < clean["score"], (
            f"Missing score ({missing['score']}) should be < clean ({clean['score']})"
        )
        assert missing["missing_rate"] > 0.3  # ~50% missing

    def test_deterministic_same_input(self):
        """Ayni giris ayni skoru vermeli (deterministik)."""
        r1 = data_quality_score(_clean_df())
        r2 = data_quality_score(_clean_df())
        assert r1["score"] == r2["score"]
        assert r1["missing_rate"] == r2["missing_rate"]

    def test_score_range(self):
        """Skor her zaman [0, 100] araliginda olmali."""
        df = _missing_df()
        result = data_quality_score(df)
        assert 0.0 <= result["score"] <= 100.0

    def test_duplicate_detection(self):
        """Tekrarli satirlar duplicate_rate'i artirmali."""
        result = data_quality_score(_duplicate_df())
        # 5 satir, 2 tekrar → duplicate_rate = 2/5 = 0.4
        assert result["duplicate_rate"] == 0.4, f"Expected 0.4, got {result['duplicate_rate']}"
        assert result["score"] < 100.0

    def test_outlier_detection(self):
        """Aykiri degerler outlier_rate'i artirmali."""
        result = data_quality_score(_outlier_df())
        # outlier_var sutununda 2/10 aykiri deger var
        assert result["outlier_rate"] > 0.0
        assert result["outlier_rate"] <= 1.0

    def test_constant_column_detection(self):
        """Sabit sutunlar constant_columns sayisina yansimali."""
        result = data_quality_score(_constant_col_df())
        # sabit_1 (hepsi 5) ve sabit_null (hepsi NaN) → 2 sabit sutun
        assert result["constant_columns"] == 2, (
            f"Expected 2 constant columns, got {result['constant_columns']}"
        )
        assert result["score"] < 100.0

    def test_empty_df(self):
        """Bos DataFrame hata vermemeli."""
        result = data_quality_score(_empty_df())
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 100.0

    def test_no_numeric_columns(self):
        """Sayisal sutun yoksa outlier_rate 0 olmali."""
        df = pd.DataFrame({"metin": ["a", "b", "c"]})
        result = data_quality_score(df)
        assert result["outlier_rate"] == 0.0
        assert result["score"] == 100.0  # Temiz veri

    def test_single_column_df(self):
        """Tek sutunlu DataFrame calismali."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        result = data_quality_score(df)
        assert result["score"] == 100.0  # Eksiksiz, tekrarsiz, aykirisiz, sabit degil

    def test_iqr_zero_column(self):
        """IQR=0 olan sutunlar icin outlier_rate 0 olmali."""
        df = pd.DataFrame({"sabit": [10, 10, 10, 10], "degisen": [1, 2, 3, 4]})
        result = data_quality_score(df)
        # IQR=0 icin outlier yok sayilir
        # "sabit" sutunu constant degil cunku nan yok ve nunique > 1... wait
        # "sabit" sutununda hepsi 10, nunique=1 → constant_columns=1
        # degisen: 1,2,3,4 → outlier check: Q1=1.5, Q3=3.5, IQR=2, fences: -1.5 to 6.5 → no outliers
        assert result["outlier_rate"] == 0.0
        assert result["constant_columns"] == 1

    def test_components_match_score(self):
        """components kaybi toplami skorla tutarli olmali."""
        result = data_quality_score(_missing_df())
        comp = result["components"]
        total_penalty = (
            comp["missing_penalty"]
            + comp["duplicate_penalty"]
            + comp["outlier_penalty"]
            + comp["constant_penalty"]
        )
        expected_score = round(max(0.0, min(100.0, 100.0 - total_penalty)), 2)
        assert abs(result["score"] - expected_score) < 0.01, (
            f"Score {result['score']} != expected {expected_score}"
        )
