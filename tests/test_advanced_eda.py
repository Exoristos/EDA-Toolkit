"""Phase 3: Advanced EDA tests for eda_toolkit."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from eda_toolkit.loader import load_dataframe
from eda_toolkit.eda.correlation import correlation_matrix, categorical_summary
from eda_toolkit.eda.advanced import (
    detect_outliers,
    run_hypothesis_tests,
    cluster,
    run_advanced_eda,
)

FIXTURES_ADV = os.path.join(os.path.dirname(__file__), "fixtures", "advanced")


# ── Fixture helpers ───────────────────────────────────────────────────────────

def _load_clusters() -> pd.DataFrame:
    return load_dataframe(os.path.join(FIXTURES_ADV, "numeric_clusters.csv"))


def _load_degenerate() -> pd.DataFrame:
    return load_dataframe(os.path.join(FIXTURES_ADV, "degenerate.csv"))


# ═══════════════════════════════════════════════════════════════════════════════
# correlation.py tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorrelationMatrix:
    def test_square_numeric_matrix(self):
        df = _load_clusters()
        mat = correlation_matrix(df)
        assert isinstance(mat, pd.DataFrame)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        assert list(mat.index) == numeric_cols
        assert list(mat.columns) == numeric_cols
        # Diagonal should be 1.0
        for col in numeric_cols:
            assert mat.loc[col, col] == pytest.approx(1.0)

    def test_empty_on_no_numeric(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        mat = correlation_matrix(df)
        assert mat.empty

    def test_returns_dataframe_always(self):
        mat = correlation_matrix(pd.DataFrame())
        assert isinstance(mat, pd.DataFrame)
        assert mat.empty


class TestCategoricalSummary:
    def test_returns_dict_for_object_cols(self):
        df = _load_clusters()
        summary = categorical_summary(df)
        assert isinstance(summary, dict)
        # cat_col is object
        assert "cat_col" in summary
        assert "counts" in summary["cat_col"]
        assert "n_unique" in summary["cat_col"]
        assert summary["cat_col"]["n_unique"] == 4  # A, B, C, D

    def test_no_object_cols_returns_empty(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        summary = categorical_summary(df)
        assert summary == {}


# ═══════════════════════════════════════════════════════════════════════════════
# advanced.py tests — detect_outliers
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectOutliers:
    def test_flags_injected_outliers(self):
        df = _load_clusters()
        result = detect_outliers(df)
        assert "skor" in result
        # The first 10 rows were injected with extreme skor values
        assert result["skor"]["count"] > 0
        assert result["skor"]["rate"] > 0.0

    def test_returns_per_column_dict(self):
        df = _load_clusters()
        result = detect_outliers(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            assert col in result
            assert "count" in result[col]
            assert "rate" in result[col]
            assert "lower" in result[col]
            assert "upper" in result[col]

    def test_handles_constant_column(self):
        df = _load_degenerate()
        # constant_col is all "100"
        result = detect_outliers(df)
        # detect_outliers only iterates numeric columns; constant_col loaded as int
        # single_num should be present
        assert "single_num" in result


# ═══════════════════════════════════════════════════════════════════════════════
# advanced.py tests — run_hypothesis_tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestHypothesisTests:
    def test_returns_list_with_shapiro_entries(self):
        df = _load_clusters()
        results = run_hypothesis_tests(df)
        assert isinstance(results, list)
        shapiro_tests = [r for r in results if r["test"] == "shapiro"]
        assert len(shapiro_tests) > 0
        for r in shapiro_tests:
            assert "column" in r
            assert "is_normal" in r

    def test_is_normal_reflects_distribution(self):
        """Generate normal and non-normal columns to verify is_normal flag."""
        np.random.seed(42)
        n = 200
        # Normal distribution
        normal_col = np.random.normal(100, 15, n)
        # Exponential (non-normal)
        exp_col = np.random.exponential(10, n)
        df = pd.DataFrame({
            "normal_dist": normal_col,
            "exp_dist": exp_col,
        })
        results = run_hypothesis_tests(df)
        shapiro_dict = {r["column"]: r for r in results if r["test"] == "shapiro"}
        # Normal distribution may or may not pass depending on random seed,
        # but exponential should almost certainly reject normality
        assert shapiro_dict["exp_dist"]["is_normal"] is False

    def test_constant_column_handled(self):
        df = pd.DataFrame({"x": [5] * 50, "y": np.random.normal(0, 1, 50)})
        results = run_hypothesis_tests(df)
        # x should have note about constant
        shapiro_x = [r for r in results if r["test"] == "shapiro" and r["column"] == "x"]
        assert len(shapiro_x) > 0
        assert shapiro_x[0].get("note") is not None or shapiro_x[0]["statistic"] is None

    def test_no_crash_on_tiny_data(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        results = run_hypothesis_tests(df)
        # Should not crash, even with tiny data
        assert isinstance(results, list)

    def test_pearson_pairs_returned(self):
        df = _load_clusters()
        results = run_hypothesis_tests(df)
        pearson_tests = [r for r in results if r["test"] == "pearson"]
        assert len(pearson_tests) > 0
        for r in pearson_tests:
            assert "columns" in r
            assert len(r["columns"]) == 2
            assert "statistic" in r
            assert "significant" in r


# ═══════════════════════════════════════════════════════════════════════════════
# advanced.py tests — cluster
# ═══════════════════════════════════════════════════════════════════════════════

class TestCluster:
    def test_cluster_returns_labels_matching_row_count(self):
        df = _load_clusters()
        result = cluster(df, max_k=4)
        assert result["performed"] is True
        assert len(result["labels"]) == len(df)
        assert result["k"] is not None

    def test_degenerate_returns_performed_false(self):
        df = _load_degenerate()
        result = cluster(df, max_k=4)
        assert result["performed"] is False
        assert "reason" in result
        assert result["k"] is None

    def test_single_numeric_col_fails(self):
        df = pd.DataFrame({"x": np.random.randn(20)})
        result = cluster(df, max_k=4)
        assert result["performed"] is False

    def test_too_few_rows_fails(self):
        df = pd.DataFrame({"x": np.random.randn(5), "y": np.random.randn(5)})
        result = cluster(df, max_k=4)
        assert result["performed"] is False

    def test_anomaly_series_is_bool(self):
        df = _load_clusters()
        result = cluster(df, max_k=4)
        assert result["anomaly"].dtype == bool
        assert len(result["anomaly"]) == len(df)

    def test_silhouette_is_float_or_none(self):
        df = _load_clusters()
        result = cluster(df, max_k=3)
        assert isinstance(result["silhouette"], (float, type(None)))


# ═══════════════════════════════════════════════════════════════════════════════
# advanced.py tests — run_advanced_eda orchestration
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunAdvancedEda:
    def test_returns_dict_with_5_keys(self):
        df = _load_clusters()
        result = run_advanced_eda(df)
        assert isinstance(result, dict)
        expected_keys = {
            "correlation_matrix",
            "categorical_summary",
            "outliers",
            "hypothesis_tests",
            "clustering",
        }
        assert set(result.keys()) == expected_keys

    def test_shapes_consistent(self):
        df = _load_clusters()
        result = run_advanced_eda(df)

        # correlation_matrix: square, numeric cols
        mat = result["correlation_matrix"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            assert mat.shape[0] == mat.shape[1] == len(numeric_cols)

        # outliers: one entry per numeric col
        outliers = result["outliers"]
        assert len(outliers) == len(numeric_cols)

        # hypothesis_tests: list
        assert isinstance(result["hypothesis_tests"], list)

        # clustering: has performed, labels length == df length
        clust = result["clustering"]
        assert "performed" in clust
        assert len(clust["labels"]) == len(df)
        assert len(clust["anomaly"]) == len(df)

    def test_run_on_degenerate_does_not_crash(self):
        df = _load_degenerate()
        result = run_advanced_eda(df)
        assert isinstance(result, dict)
        assert "correlation_matrix" in result
        assert "outliers" in result
        assert "hypothesis_tests" in result
        assert "clustering" in result
        assert result["clustering"]["performed"] is False
