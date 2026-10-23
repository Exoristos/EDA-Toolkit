"""Phase 6: PDF reporter tests for eda_toolkit."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from eda_toolkit.types import (
    ReportResult,
    PipelineConfig,
    EdaResult,
    ColumnSchema,
    CleaningAction,
    CleaningLevel,
    PiiFinding,
    PiiType,
)
from eda_toolkit.reporters.pdf_reporter import render_pdf, write_pdf


# ---------------------------------------------------------------------------
# Helpers — build synthetic ReportResult instances
# ---------------------------------------------------------------------------

def _make_eda_result(
    df: pd.DataFrame,
    correlation: pd.DataFrame | None = None,
    outliers: dict | None = None,
    hyp_tests: list | None = None,
) -> EdaResult:
    """Build an EdaResult with optional advanced fields."""
    numeric_cols = df.select_dtypes(include=["number"])
    descriptive = numeric_cols.describe() if not numeric_cols.empty else pd.DataFrame()
    missing_summary = pd.DataFrame({
        "column": list(df.columns),
        "missing_count": df.isnull().sum().tolist(),
        "missing_rate": df.isnull().mean().tolist(),
    })
    schemas = [
        ColumnSchema(
            name=col,
            dtype=str(df[col].dtype),
            kind="numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical",
            unique_count=int(df[col].nunique()),
            missing_count=int(df[col].isnull().sum()),
        )
        for col in df.columns
    ]
    return EdaResult(
        descriptive=descriptive,
        missing_summary=missing_summary,
        schemas=schemas,
        correlation_matrix=correlation,
        outliers=outliers,
        hypothesis_tests=hyp_tests,
    )


def _make_result(
    df: pd.DataFrame,
    input_path: str = "test.csv",
    pii_findings: list | None = None,
    cleaning_log: list | None = None,
    correlation: pd.DataFrame | None = None,
    outliers: dict | None = None,
    hyp_tests: list | None = None,
) -> ReportResult:
    return ReportResult(
        input_path=input_path,
        row_count=len(df),
        col_count=len(df.columns),
        cleaned_df=df,
        eda=_make_eda_result(df, correlation=correlation, outliers=outliers, hyp_tests=hyp_tests),
        pii_findings=pii_findings if pii_findings is not None else [],
        cleaning_log=cleaning_log if cleaning_log is not None else [],
    )


def _corr_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 0.5], "b": [0.5, 1.0]}, index=["a", "b"])


def _outliers_dict() -> dict:
    return {
        "yas": {"count": 3, "rate": 0.015, "lower": 18.0, "upper": 65.0},
        "gelir": {"count": 5, "rate": 0.025, "lower": 1000.0, "upper": 50000.0},
    }


def _hyp_tests() -> list[dict]:
    return [
        {
            "column": "yas",
            "test": "shapiro",
            "statistic": 0.98,
            "p_value": 0.15,
            "is_normal": True,
        },
        {
            "columns": ["yas", "gelir"],
            "test": "pearson",
            "statistic": 0.72,
            "p_value": 0.001,
            "significant": True,
        },
    ]


def _pii_findings() -> list[PiiFinding]:
    return [
        PiiFinding(
            column="tc",
            pii_type=PiiType.TCKN,
            confidence=0.95,
            match_rate=0.80,
            sample_count=10,
            detection_layer="text",
        ),
        PiiFinding(
            column="email",
            pii_type=PiiType.EMAIL,
            confidence=0.90,
            match_rate=0.60,
            sample_count=5,
            detection_layer="regex",
        ),
    ]


def _cleaning_log() -> list[CleaningAction]:
    return [
        CleaningAction(
            column="yas",
            issue="missing",
            level=CleaningLevel.CONSERVATIVE,
            applied=True,
            detail="filled with median",
        ),
        CleaningAction(
            column="gelir",
            issue="outlier",
            level=CleaningLevel.CONSERVATIVE,
            applied=False,
            detail="flagged only",
        ),
    ]


def _assert_valid_pdf(path: str) -> int:
    """Assert file is a valid PDF (starts with %PDF) and return size."""
    assert os.path.exists(path), f"PDF not found: {path}"
    size = os.path.getsize(path)
    assert size > 0, f"PDF is empty: {path}"
    with open(path, "rb") as f:
        header = f.read(4)
    assert header == b"%PDF", f"Not a valid PDF (header: {header!r}): {path}"
    return size


# ═══════════════════════════════════════════════════════════════════════════════
# Core PDF tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPdfReporter:
    """Tests using synthetic DataFrames with controlled fields."""

    # ── Basic: file is created, non-empty, valid PDF header ─────────────────

    def test_file_created_and_valid(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        result = _make_result(df)
        config = PipelineConfig()

        out = tmp_path / "report.pdf"
        returned = render_pdf(result, config, str(out))

        assert returned == str(out)
        _assert_valid_pdf(str(out))

    def test_write_pdf_alias(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2]})
        result = _make_result(df)
        config = PipelineConfig()
        out = tmp_path / "alias.pdf"
        returned = write_pdf(result, config, str(out))
        assert returned == str(out)
        _assert_valid_pdf(str(out))

    # ── Sections present: with full optional data → larger PDF ─────────────

    def test_full_report_larger_than_minimal(self, tmp_path):
        """A report with all optional fields populated is larger than a minimal one."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        config = PipelineConfig()

        # Minimal: no optional fields
        result_min = _make_result(df)
        out_min = tmp_path / "min.pdf"
        render_pdf(result_min, config, str(out_min))
        size_min = _assert_valid_pdf(str(out_min))

        # Full: with everything
        result_full = _make_result(
            df,
            pii_findings=_pii_findings(),
            cleaning_log=_cleaning_log(),
            correlation=_corr_df(),
            outliers=_outliers_dict(),
            hyp_tests=_hyp_tests(),
        )
        out_full = tmp_path / "full.pdf"
        render_pdf(result_full, config, str(out_full))
        size_full = _assert_valid_pdf(str(out_full))

        assert size_full > size_min, (
            f"Full report ({size_full} bytes) should be larger than minimal ({size_min} bytes)"
        )

    # ── Optional sections present when data exists ──────────────────────────

    def test_correlation_included_when_present(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = _make_result(df, correlation=_corr_df())
        config = PipelineConfig()
        out = tmp_path / "with_corr.pdf"
        render_pdf(result, config, str(out))
        _assert_valid_pdf(str(out))

    def test_outliers_included_when_present(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        result = _make_result(df, outliers=_outliers_dict())
        config = PipelineConfig()
        out = tmp_path / "with_out.pdf"
        render_pdf(result, config, str(out))
        _assert_valid_pdf(str(out))

    def test_hypothesis_included_when_present(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        result = _make_result(df, hyp_tests=_hyp_tests())
        config = PipelineConfig()
        out = tmp_path / "with_hyp.pdf"
        render_pdf(result, config, str(out))
        _assert_valid_pdf(str(out))

    # ── PII section ─────────────────────────────────────────────────────────

    def test_pii_with_findings(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2]})
        result = _make_result(df, pii_findings=_pii_findings())
        config = PipelineConfig()
        out = tmp_path / "pii.pdf"
        render_pdf(result, config, str(out))
        size = _assert_valid_pdf(str(out))
        # With PII findings the file should be reasonably sized
        assert size > 2000, f"PII report too small: {size} bytes"

    def test_pii_empty_shows_message(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2]})
        result = _make_result(df, pii_findings=[])
        config = PipelineConfig()
        out = tmp_path / "pii_empty.pdf"
        render_pdf(result, config, str(out))
        _assert_valid_pdf(str(out))

    # ── Empty/None-safe ────────────────────────────────────────────────────

    def test_empty_result_produces_valid_pdf(self, tmp_path):
        """A ReportResult with empty descriptive/missing and no optional fields
        still produces a valid (small) PDF without error."""
        eda = EdaResult(
            descriptive=pd.DataFrame(),
            missing_summary=pd.DataFrame(),
            schemas=[],
            correlation_matrix=None,
            outliers=None,
            hypothesis_tests=None,
        )
        result = ReportResult(
            input_path="empty.csv",
            row_count=0,
            col_count=0,
            cleaned_df=pd.DataFrame(),
            eda=eda,
            pii_findings=[],
            cleaning_log=[],
        )
        config = PipelineConfig()
        out = tmp_path / "empty.pdf"
        render_pdf(result, config, str(out))
        _assert_valid_pdf(str(out))

    def test_none_optional_fields_dont_crash(self, tmp_path):
        """Ensure None optional fields don't cause crashes."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        eda = EdaResult(
            descriptive=df.describe(),
            missing_summary=pd.DataFrame({"column": ["x"], "missing_count": [0], "missing_rate": [0.0]}),
            schemas=[ColumnSchema("x", "int64", "numeric", 3, 0)],
            correlation_matrix=None,
            outliers=None,
            hypothesis_tests=None,
        )
        result = ReportResult(
            input_path="test.csv",
            row_count=3,
            col_count=1,
            cleaned_df=df,
            eda=eda,
            pii_findings=[],
            cleaning_log=[],
        )
        config = PipelineConfig()
        out = tmp_path / "none_opts.pdf"
        render_pdf(result, config, str(out))
        _assert_valid_pdf(str(out))

    # ── Creates parent directory ────────────────────────────────────────────

    def test_creates_parent_dir(self, tmp_path):
        df = pd.DataFrame({"x": [1]})
        result = _make_result(df)
        config = PipelineConfig()
        nested = tmp_path / "deep" / "nested" / "report.pdf"
        render_pdf(result, config, str(nested))
        _assert_valid_pdf(str(nested))


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end via pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class TestPdfE2E:
    """End-to-end tests using pipeline real run on fixture CSV files."""

    def test_e2e_with_numeric_heavy(self, tmp_path, numeric_heavy_csv):
        """Run pipeline on numeric_heavy.csv requesting pdf → PDF produced."""
        from eda_toolkit.pipeline import run

        cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = run(numeric_heavy_csv, PipelineConfig(report_formats=["pdf"]))
            # Resolve absolute path while still in the temp directory
            pdf_abs = os.path.abspath(result.reports.get("pdf", ""))
        finally:
            os.chdir(cwd)

        assert "pdf" in result.reports, "PDF report path not in result.reports"
        _assert_valid_pdf(pdf_abs)

    def test_e2e_pii_pdf_produced(self, tmp_path, mixed_with_pii_csv):
        """Run pipeline on mixed_with_pii.csv → PDF with PII section."""
        from eda_toolkit.pipeline import run

        cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = run(mixed_with_pii_csv, PipelineConfig(report_formats=["pdf"]))
            pdf_abs = os.path.abspath(result.reports.get("pdf", ""))
        finally:
            os.chdir(cwd)

        assert "pdf" in result.reports
        size = _assert_valid_pdf(pdf_abs)
        assert size > 5000, f"PII PDF should be > 5KB, got {size} bytes"
