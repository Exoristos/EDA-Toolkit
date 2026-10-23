"""Phase 5: Excel reporter tests for eda_toolkit."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

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
from eda_toolkit.reporters.excel_reporter import render_excel, write_excel


# ── Helper: build a minimal ReportResult ─────────────────────────────────────

def _make_eda_result(df: pd.DataFrame) -> EdaResult:
    """Build an EdaResult with all non-advanced fields populated."""
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
    return EdaResult(descriptive=descriptive, missing_summary=missing_summary, schemas=schemas)


def _make_result(
    df: pd.DataFrame,
    input_path: str = "test.csv",
    pii_findings: list | None = None,
    cleaning_log: list | None = None,
    correlation_matrix: pd.DataFrame | None = None,
) -> ReportResult:
    return ReportResult(
        input_path=input_path,
        row_count=len(df),
        col_count=len(df.columns),
        cleaned_df=df,
        eda=_make_eda_result(df),
        pii_findings=pii_findings if pii_findings is not None else [],
        cleaning_log=cleaning_log if cleaning_log is not None else [],
    )


# ── Fixture helpers ───────────────────────────────────────────────────────────

def _corr_df() -> pd.DataFrame:
    """Small 2x2 correlation matrix for advanced-field tests."""
    return pd.DataFrame({"a": [1.0, 0.5], "b": [0.5, 1.0]}, index=["a", "b"])


def _pii_findings() -> list[PiiFinding]:
    return [
        PiiFinding(column="tc", pii_type=PiiType.TCKN, confidence=0.95, match_rate=0.80, sample_count=10, detection_layer="text"),
        PiiFinding(column="email", pii_type=PiiType.EMAIL, confidence=0.90, match_rate=0.60, sample_count=5, detection_layer="regex"),
    ]


def _cleaning_log() -> list[CleaningAction]:
    return [
        CleaningAction(column="yas", issue="missing", level=CleaningLevel.CONSERVATIVE, applied=True, detail="filled with median"),
        CleaningAction(column="gelir", issue="outlier", level=CleaningLevel.CONSERVATIVE, applied=False, detail="flagged only"),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Core behavior
# ═══════════════════════════════════════════════════════════════════════════════

class TestExcelReporter:
    """Tests using synthetic DataFrames with controlled fields."""

    # ── Basic: file is created and non-empty ──────────────────────────────

    def test_file_created_and_nonempty(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        result = _make_result(df)
        config = PipelineConfig()

        out = tmp_path / "report.xlsx"
        returned = render_excel(result, config, str(out))

        assert returned == str(out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_write_excel_alias(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2]})
        result = _make_result(df)
        config = PipelineConfig()
        out = tmp_path / "alias.xlsx"
        returned = write_excel(result, config, str(out))
        assert returned == str(out)
        assert out.exists()

    # ── Sheet presence: all core sheets ───────────────────────────────────

    def test_all_core_sheets_present(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        result = _make_result(df, pii_findings=_pii_findings(), cleaning_log=_cleaning_log())
        # Also set correlation to None → Korelasyon should be absent
        config = PipelineConfig()

        out = tmp_path / "core.xlsx"
        render_excel(result, config, str(out))

        wb = load_workbook(str(out))
        sheet_names = wb.sheetnames

        expected = {"Ozet", "Ham_Veri", "Tanimlayici_Istatistik", "Eksik_Veri", "Temizleme_Logu", "PII_Uyarilari"}
        for name in expected:
            assert name in sheet_names, f"Expected sheet '{name}' not found in {sheet_names}"

        # Korelasyon should NOT be present when correlation_matrix is None
        assert "Korelasyon" not in sheet_names

    # ── Korelasyon present when set ───────────────────────────────────────

    def test_korelasyon_present_when_set(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = _make_result(df, correlation_matrix=_corr_df())
        result.eda.correlation_matrix = _corr_df()
        config = PipelineConfig()
        out = tmp_path / "with_corr.xlsx"
        render_excel(result, config, str(out))
        wb = load_workbook(str(out))
        assert "Korelasyon" in wb.sheetnames

    def test_korelasyon_absent_when_none(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = _make_result(df)
        result.eda.correlation_matrix = None
        config = PipelineConfig()
        out = tmp_path / "no_corr.xlsx"
        render_excel(result, config, str(out))
        wb = load_workbook(str(out))
        assert "Korelasyon" not in wb.sheetnames

    # ── PII_Uyarilari always present ──────────────────────────────────────

    def test_pii_sheet_with_data(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2]})
        result = _make_result(df, pii_findings=_pii_findings())
        config = PipelineConfig()
        out = tmp_path / "pii.xlsx"
        render_excel(result, config, str(out))
        wb = load_workbook(str(out))
        assert "PII_Uyarilari" in wb.sheetnames
        ws = wb["PII_Uyarilari"]
        # Header row + 2 data rows
        assert ws.max_row >= 3  # header + 2 findings

    def test_pii_sheet_header_only_when_empty(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2]})
        result = _make_result(df, pii_findings=[])
        config = PipelineConfig()
        out = tmp_path / "pii_empty.xlsx"
        render_excel(result, config, str(out))
        wb = load_workbook(str(out))
        assert "PII_Uyarilari" in wb.sheetnames
        ws = wb["PII_Uyarilari"]
        # Only header row
        assert ws.max_row == 1
        assert ws.max_column >= 4  # has header columns

    # ── Temizleme_Logu always present ─────────────────────────────────────

    def test_cleaning_log_with_data(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2]})
        result = _make_result(df, cleaning_log=_cleaning_log())
        config = PipelineConfig()
        out = tmp_path / "clean.xlsx"
        render_excel(result, config, str(out))
        wb = load_workbook(str(out))
        assert "Temizleme_Logu" in wb.sheetnames
        ws = wb["Temizleme_Logu"]
        assert ws.max_row >= 3  # header + 2 actions

    def test_cleaning_log_header_only_when_empty(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2]})
        result = _make_result(df, cleaning_log=[])
        config = PipelineConfig()
        out = tmp_path / "clean_empty.xlsx"
        render_excel(result, config, str(out))
        wb = load_workbook(str(out))
        assert "Temizleme_Logu" in wb.sheetnames
        ws = wb["Temizleme_Logu"]
        assert ws.max_row == 1  # only header

    # ── Sheet name constraints ────────────────────────────────────────────

    def test_sheet_names_within_31_chars(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2]})
        result = _make_result(df)
        config = PipelineConfig()
        out = tmp_path / "names.xlsx"
        render_excel(result, config, str(out))
        wb = load_workbook(str(out))
        for name in wb.sheetnames:
            assert len(name) <= 31, f"Sheet name '{name}' exceeds 31 chars"

    def test_sheet_names_are_unique(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2]})
        result = _make_result(df)
        config = PipelineConfig()
        out = tmp_path / "unique.xlsx"
        render_excel(result, config, str(out))
        wb = load_workbook(str(out))
        assert len(wb.sheetnames) == len(set(wb.sheetnames)), f"Duplicate sheet names: {wb.sheetnames}"

    # ── Ozet content ──────────────────────────────────────────────────────

    def test_ozet_row_count_matches(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})
        result = _make_result(df)
        config = PipelineConfig()
        out = tmp_path / "ozet.xlsx"
        render_excel(result, config, str(out))
        wb = load_workbook(str(out))
        ws = wb["Ozet"]
        # Find row that contains "Satir Sayisi" and check its Değer
        ozet_dict = {}
        for row in ws.iter_rows(min_row=1, max_row=ws.max_row, values_only=True):
            if row[0] and row[1] is not None:
                ozet_dict[str(row[0])] = row[1]
        assert ozet_dict.get("Satir Sayisi") == 5

    # ── Formatting basics ─────────────────────────────────────────────────

    def test_header_is_bold(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _make_result(df)
        config = PipelineConfig()
        out = tmp_path / "bold.xlsx"
        render_excel(result, config, str(out))
        wb = load_workbook(str(out))
        for ws in wb.worksheets:
            first_cell = ws.cell(row=1, column=1)
            if first_cell.value is not None:
                assert first_cell.font.bold is True, f"Header not bold in {ws.title}"

    def test_first_row_frozen(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2]})
        result = _make_result(df)
        config = PipelineConfig()
        out = tmp_path / "frozen.xlsx"
        render_excel(result, config, str(out))
        wb = load_workbook(str(out))
        for ws in wb.worksheets:
            assert ws.freeze_panes is not None, f"No freeze panes in {ws.title}"
            # "A2" means row 1 is frozen
            assert ws.freeze_panes == "A2", f"Expected A2 freeze got {ws.freeze_panes} in {ws.title}"

    # ── Creates parent directory ──────────────────────────────────────────

    def test_creates_parent_dir(self, tmp_path):
        df = pd.DataFrame({"x": [1]})
        result = _make_result(df)
        config = PipelineConfig()
        nested = tmp_path / "deep" / "nested" / "report.xlsx"
        render_excel(result, config, str(nested))
        assert nested.exists()
        assert nested.stat().st_size > 0


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end via pipeline on numeric_heavy.csv
# ═══════════════════════════════════════════════════════════════════════════════

class TestExcelE2E:
    def test_e2e_with_numeric_heavy(self, tmp_path, numeric_heavy_csv):
        """Load numeric_heavy.csv, run pipeline, render_excel, verify Ozet."""
        from eda_toolkit.pipeline import run

        # Run pipeline with default config (HTML by default, so we get cleaned data)
        import os as _os
        cwd = _os.getcwd()
        try:
            _os.chdir(str(tmp_path))
            result = run(numeric_heavy_csv)
        finally:
            _os.chdir(cwd)

        # Now render Excel
        config = PipelineConfig(report_formats=["excel"])
        out_xlsx = str(tmp_path / "numeric_heavy_report.xlsx")
        render_excel(result, config, out_xlsx)

        assert os.path.exists(out_xlsx)
        assert os.path.getsize(out_xlsx) > 0

        # Open and verify Ozet row_count
        wb = load_workbook(out_xlsx)
        assert "Ozet" in wb.sheetnames

        ws = wb["Ozet"]
        ozet = {}
        for row in ws.iter_rows(min_row=1, max_row=ws.max_row, values_only=True):
            if row[0] and row[1] is not None:
                ozet[str(row[0])] = row[1]

        assert ozet.get("Satir Sayisi") == 200
        assert ozet.get("Sutun Sayisi") == 7
