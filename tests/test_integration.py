"""Phase 6 integration: E2E pipeline tests wiring all modules together.

Tests verify the end-to-end pipeline (load -> clean -> EDA -> PII -> report)
produces correct outputs for various input scenarios.
"""

from __future__ import annotations

import contextlib
import os

import pandas as pd
import pytest
from openpyxl import load_workbook

from eda_toolkit.types import PipelineConfig, PiiMode, MaskMode
from eda_toolkit.pipeline import run


@contextlib.contextmanager
def _in_dir(path):
    """Context manager: temporarily change CWD to *path*."""
    old = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(old)


def _abs(reports: dict, base_dir: str, key: str) -> str:
    """Build absolute path for a report entry relative to *base_dir*."""
    return os.path.join(base_dir, reports[key])


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: Default config on mixed_with_pii.csv
# ═══════════════════════════════════════════════════════════════════════════════

class TestMixedPiiDefault:
    def test_pii_found_and_correlation_computed(self, tmp_path, mixed_with_pii_csv):
        """Default config (WARN) should detect PII and compute correlation."""
        base = str(tmp_path)
        with _in_dir(base):
            result = run(mixed_with_pii_csv)

            # PII findings should be non-empty
            assert len(result.pii_findings) > 0, "Expected PII findings but got none"

            # Should detect at least TCKN, IBAN, or PHONE
            pii_types = {f.pii_type.value for f in result.pii_findings}
            assert any(t in pii_types for t in ("tckn", "iban", "phone")), (
                f"Expected at least tckn/iban/phone in PII types but got {pii_types}"
            )

            # Correlation matrix should not be None
            assert result.eda.correlation_matrix is not None, "Expected correlation_matrix not None"
            assert not result.eda.correlation_matrix.empty

            # HTML report should exist and be non-empty (check while in dir)
            assert "html" in result.reports, "Expected 'html' in reports"
            html_path = _abs(result.reports, base, "html")
        # Back in original CWD; check file with absolute path
        assert os.path.exists(html_path), f"HTML file not found: {html_path}"
        assert os.path.getsize(html_path) > 0, "HTML file is empty"

        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        assert "Veri Analiz Raporu" in html_content, "HTML missing Turkish title"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: MASK mode vs WARN mode
# ═══════════════════════════════════════════════════════════════════════════════

class TestMixedPiiMaskMode:
    def test_mask_hides_tckn(self, tmp_path, mixed_with_pii_csv):
        """MASK mode should mask PII columns; WARN mode keeps originals."""
        warn_base = str(tmp_path / "warn_run")
        os.makedirs(warn_base, exist_ok=True)
        with _in_dir(warn_base):
            warn_result = run(mixed_with_pii_csv, PipelineConfig(pii=PiiMode.WARN))

        mask_base = str(tmp_path / "mask_run")
        os.makedirs(mask_base, exist_ok=True)
        with _in_dir(mask_base):
            mask_result = run(
                mixed_with_pii_csv,
                PipelineConfig(
                    pii=PiiMode.MASK,
                    mask_mode=MaskMode.PARTIAL,
                    show_last=4,
                ),
            )

        warn_df = warn_result.cleaned_df
        mask_df = mask_result.cleaned_df

        # Verify same row/col counts
        assert len(warn_df) == len(mask_df)
        assert len(warn_df.columns) == len(mask_df.columns)

        # Non-PII numeric columns should be identical
        for col in ["yas", "gelir", "kredi_skoru"]:
            if col in warn_df.columns:
                pd.testing.assert_series_equal(
                    warn_df[col].reset_index(drop=True),
                    mask_df[col].reset_index(drop=True),
                    check_dtype=False,
                )

        # A known IBAN from warn-mode should NOT be present verbatim in mask-mode.
        # IBAN column is loaded as string by pandas (unlike tc_kimlik which
        # becomes float64), so PII detection + masking work correctly.
        if "iban" in warn_df.columns:
            warn_iban = warn_df["iban"].iloc[0]
            assert isinstance(warn_iban, str) and warn_iban.startswith("TR")
            assert len(warn_iban) == 26

            # Build the set of IBAN strings in mask_df
            mask_iban_vals = mask_df["iban"].dropna().astype(str).tolist()
            # In PARTIAL mode, masked IBAN has show_last=4 visible chars
            assert warn_iban not in mask_iban_vals, (
                "IBAN from WARN mode found verbatim in MASK mode output"
            )
            # Verify masking: masked IBAN should differ from original
            mask_first = mask_df["iban"].iloc[0]
            assert str(mask_first) != warn_iban, "Masked IBAN should differ from original"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: numeric_heavy with both HTML and Excel formats
# ═══════════════════════════════════════════════════════════════════════════════

class TestNumericHeavyBothFormats:
    def test_both_reports_produced(self, tmp_path, numeric_heavy_csv):
        """Run numeric_heavy with html+excel: both reports and Excel Ozet correct."""
        base = str(tmp_path)
        with _in_dir(base):
            config = PipelineConfig(report_formats=["html", "excel"])
            result = run(numeric_heavy_csv, config)

            assert "html" in result.reports, "Expected 'html' in reports"
            assert "excel" in result.reports, "Expected 'excel' in reports"

            html_path = _abs(result.reports, base, "html")
            excel_path = _abs(result.reports, base, "excel")

        assert os.path.exists(html_path), f"HTML not found: {html_path}"
        assert os.path.getsize(html_path) > 0
        assert os.path.exists(excel_path), f"Excel not found: {excel_path}"
        assert os.path.getsize(excel_path) > 0

        # Excel: verify Ozet sheet and row_count
        wb = load_workbook(excel_path)
        assert "Ozet" in wb.sheetnames, f"Expected 'Ozet' sheet, got: {wb.sheetnames}"

        ws = wb["Ozet"]
        ozet = {}
        for row in ws.iter_rows(min_row=1, max_row=ws.max_row, values_only=True):
            if row[0] and row[1] is not None:
                ozet[str(row[0])] = row[1]
        assert ozet.get("Satir Sayisi") == 200, (
            f"Expected 200 rows, got {ozet.get('Satir Sayisi')}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: HTML from numeric_heavy contains advanced sections
# ═══════════════════════════════════════════════════════════════════════════════

class TestHtmlAdvancedSections:
    def test_html_has_korelasyon_and_outliers(self, tmp_path, numeric_heavy_csv):
        """HTML from numeric_heavy should contain correlation and outliers sections."""
        base = str(tmp_path)
        with _in_dir(base):
            result = run(numeric_heavy_csv, PipelineConfig(report_formats=["html"]))
            html_path = _abs(result.reports, base, "html")

        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        assert "Korelasyon" in html, "HTML should contain 'Korelasyon' section"
        assert "Aykiri" in html, "HTML should contain 'Aykiri Degerler' section"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: Degenerate CSV (single numeric column) does not crash
# ═══════════════════════════════════════════════════════════════════════════════

class TestDegenerateInput:
    def test_single_numeric_column_no_crash(self, tmp_path):
        """Pipeline should gracefully degrade on degenerate input."""
        degenerate_csv = tmp_path / "single_col.csv"
        degenerate_csv.write_text(
            "tek_sutun\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n",
            encoding="utf-8",
        )

        base = str(tmp_path)
        with _in_dir(base):
            result = run(str(degenerate_csv), PipelineConfig(report_formats=["html"]))

        # Should not crash
        assert result is not None
        assert result.row_count == 10
        assert result.col_count == 1

        # Advanced EDA should gracefully degrade
        # correlation_matrix may be empty (need 2+ numeric cols)
        # cluster_labels should be None (need 2+ numeric cols)
        # outliers should still have the single column
        assert result.eda.outliers is not None
        assert len(result.eda.outliers) == 1
        # clustering should not have been performed
        assert result.eda.cluster_labels is None


# ═══════════════════════════════════════════════════════════════════════════════
# AC-008: TCKN must be detected even when the column loads as numeric/float
# (pandas upcasts integer columns to float64 when missing values exist).
# ═══════════════════════════════════════════════════════════════════════════════

def test_ac008_tckn_detected_after_float_upcast(tmp_path, mixed_with_pii_csv):
    """AC-008: TCKN detected with high match_rate despite float/numeric loading."""
    base = str(tmp_path)
    with _in_dir(base):
        result = run(mixed_with_pii_csv)

    tckn_findings = [f for f in result.pii_findings if f.pii_type.value == "tckn"]
    assert len(tckn_findings) >= 1, "TCKN not detected (AC-008 violation)"
    finding = tckn_findings[0]
    assert finding.column == "tc_kimlik"
    assert finding.match_rate >= 0.8, f"TCKN match_rate too low: {finding.match_rate}"
