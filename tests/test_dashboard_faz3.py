"""Faz-3 dashboard helper unit tests.

Pure (no Streamlit runtime) tests for the helper functions added with the
quality/risk/forecast/regression UI. Inline fixtures only; this file does
NOT import or edit ``conftest.py`` (MF-1).
"""

from __future__ import annotations

import math

import pandas as pd

from eda_toolkit.dashboard.app import (
    fig_forecast,
    quality_score_to_dataframe,
    regression_result_to_dataframe,
    risk_score_to_dataframe,
)


# ── quality_score_to_dataframe ────────────────────────────────────────────────


def test_quality_score_to_dataframe_none() -> None:
    df = quality_score_to_dataframe(None)
    assert df.empty
    assert list(df.columns) == ["Bilesen", "Deger", "Skora Etkisi"]


def test_quality_score_to_dataframe_populated() -> None:
    qs = {
        "score": 72.5,
        "missing_rate": 0.05,
        "duplicate_rate": 0.1,
        "outlier_rate": 0.0,
        "constant_columns": 1,
        "components": {
            "missing_penalty": 2.0,
            "duplicate_penalty": 2.0,
            "outlier_penalty": 0.0,
            "constant_penalty": 2.0,
        },
    }
    df = quality_score_to_dataframe(qs)
    assert not df.empty
    assert df.iloc[0]["Bilesen"] == "Toplam Kalite Skoru"
    assert df.iloc[0]["Deger"] == "72.5/100"
    # Penalty cells carry a leading minus sign.
    assert df.iloc[1]["Skora Etkisi"].startswith("-")
    assert df.iloc[4]["Deger"] == 1  # constant_columns kept as int


# ── risk_score_to_dataframe ──────────────────────────────────────────────────


def test_risk_score_to_dataframe_none() -> None:
    df = risk_score_to_dataframe(None)
    assert df.empty
    assert list(df.columns) == ["Bilesen", "Katki"]


def test_risk_score_to_dataframe_with_factors_and_k() -> None:
    rs = {
        "score": 80.0,
        "factors": {
            "severity": {"score": 40.0, "max_type": "tckn"},
            "cardinality": {"score": 20.0, "max_uniqueness": 0.9,
                              "max_uniqueness_column": "tc"},
            "qi_combinability": {"score": 5.0, "qi_columns": ["sehir"],
                                  "min_group_size": 4},
            "masking_contribution": 15,
        },
        "k_anonymity": 3,
        "max_severity_type": "tckn",
        "mask_mode": None,
    }
    df = risk_score_to_dataframe(rs)
    labels = df["Bilesen"].tolist()
    assert "Toplam Risk Skoru" in labels
    assert "PII Ciddiyeti (tckn)" in labels
    assert "Kardinalite" in labels
    assert "Maskeleme Katkisi" in labels
    assert "k-anonimlik" in labels
    assert df.iloc[0]["Katki"] == "80.0/100"


# ── regression_result_to_dataframe ───────────────────────────────────────────


def test_regression_result_to_dataframe_error_yields_empty() -> None:
    df = regression_result_to_dataframe({"error": "boom"})
    assert df.empty
    assert list(df.columns) == ["Degisken", "Katsayi", "p-degeri", "VIF"]


def test_regression_result_to_dataframe_linear() -> None:
    res = {
        "kind": "linear",
        "r2": 0.9,
        "coefficients": {"intercept": 1.5, "yas": 2.25, "gelir": -0.5},
        "p_values": {"intercept": 0.001, "yas": 0.04, "gelir": 0.3},
        "vif": {"yas": 4.0, "gelir": math.nan},
        "vif_warning": [],
        "n": 100,
        "intercept": 1.5,
        "feature_names": ["yas", "gelir"],
    }
    df = regression_result_to_dataframe(res)
    assert list(df.columns) == ["Degisken", "Katsayi", "p-degeri", "VIF"]
    assert len(df) == 3
    row = df[df["Degisken"] == "gelir"].iloc[0]
    assert row["Katsayi"] == "-0.5000"
    # NaN VIF renders as "-".
    assert row["VIF"] == "-"
    # Intercept has no VIF entry -> "-".
    assert df[df["Degisken"] == "intercept"].iloc[0]["VIF"] == "-"


# ── fig_forecast ──────────────────────────────────────────────────────────────


def test_fig_forecast_returns_none_on_error() -> None:
    assert fig_forecast({"error": "x"}, "col") is None


def test_fig_forecast_returns_none_when_empty() -> None:
    assert fig_forecast({"point": [], "ci_lower": [], "ci_upper": []}, "col") is None


def test_fig_forecast_builds_with_band() -> None:
    fc = {
        "point": [10.0, 11.0, 12.0],
        "ci_lower": [9.0, 10.0, 11.0],
        "ci_upper": [11.0, 12.0, 13.0],
        "method": "arima",
        "order": "(1,0,1)",
        "horizon": 3,
        "warning": None,
    }
    history = pd.Series([1.0, 2.0, 3.0, 4.0])
    fig = fig_forecast(fc, "gelir", history)
    assert fig is not None
    # Observed (history) + forecast + CI band = 3 traces.
    assert len(fig.data) == 3
    # Title references the column.
    assert "gelir" in fig.layout.title.text
