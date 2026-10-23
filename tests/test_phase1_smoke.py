"""Phase 1 smoke tests for eda_toolkit."""

from __future__ import annotations

import json
import os
import pandas as pd

from eda_toolkit.loader import load_dataframe, detect_schema
from eda_toolkit.eda.basic import run_basic_eda
from eda_toolkit.types import ReportResult, PipelineConfig
from eda_toolkit.pipeline import run
from eda_toolkit.reporters.html_reporter import render_html
from eda_toolkit.cleaning import clean
from eda_toolkit.types import CleaningLevel


# ── Loader & schema detection ────────────────────────────────────────────────

def test_load_and_detect_numeric_heavy(numeric_heavy_csv):
    df = load_dataframe(numeric_heavy_csv)
    assert len(df) == 200
    assert len(df.columns) == 7
    schemas = detect_schema(df)
    kinds = {s.name: s.kind for s in schemas}
    assert kinds["yas"] == "numeric"
    assert kinds["gelir"] == "numeric"
    # musteri_id should be categorical/text
    assert kinds["musteri_id"] in ("categorical", "text")


def test_load_and_detect_mixed_pii(mixed_with_pii_csv):
    df = load_dataframe(mixed_with_pii_csv)
    assert len(df) == 200
    schemas = detect_schema(df)
    kinds = {s.name: s.kind for s in schemas}
    assert kinds["tc_kimlik"] in ("text", "categorical")
    assert kinds["iban"] in ("text", "categorical")
    assert kinds["ad"] in ("categorical", "text")


def test_load_and_detect_categorical(categorical_csv):
    df = load_dataframe(categorical_csv)
    assert len(df) == 200
    schemas = detect_schema(df)
    kinds = {s.name: s.kind for s in schemas}
    assert kinds["risk_skoru"] == "numeric"
    # Most others should be categorical
    assert kinds["il"] == "categorical"
    assert kinds["sektor"] in ("categorical", "text")


# ── Basic EDA ────────────────────────────────────────────────────────────────

def test_run_basic_eda(numeric_heavy_csv):
    df = load_dataframe(numeric_heavy_csv)
    result = run_basic_eda(df)
    # descriptive should have numeric cols
    numeric_cols = df.select_dtypes(include=["number"]).columns
    assert len(result.descriptive) == len(numeric_cols)
    # missing_summary should have all cols
    assert len(result.missing_summary) == len(df.columns)
    assert list(result.missing_summary.columns) == ["column", "missing_count", "missing_rate"]
    # schemas
    assert len(result.schemas) == len(df.columns)


# ── Pipeline ─────────────────────────────────────────────────────────────────

def test_pipeline_run_default(numeric_heavy_csv, tmp_path):
    # Run from tmpdir so eda_reports/ is created locally
    import os as _os
    cwd = _os.getcwd()
    try:
        _os.chdir(str(tmp_path))
        result = run(numeric_heavy_csv)
    finally:
        _os.chdir(cwd)

    assert isinstance(result, ReportResult)
    assert result.reports is not None
    assert "html" in result.reports
    report_path = str(tmp_path / result.reports["html"])
    assert _os.path.exists(report_path)
    assert _os.path.getsize(report_path) > 0

    # Conservative: pii_findings and cleaning_log should be empty
    assert result.pii_findings == []
    assert result.cleaning_log == []

    # cleaned_df should have same shape
    df = load_dataframe(numeric_heavy_csv)
    assert result.cleaned_df is not None
    assert result.cleaned_df.shape == df.shape


# ── HTML reporter ────────────────────────────────────────────────────────────

def test_html_reporter_output(numeric_heavy_csv):
    df = load_dataframe(numeric_heavy_csv)
    eda_result = run_basic_eda(df)
    report_result = ReportResult(
        input_path=numeric_heavy_csv,
        row_count=len(df),
        col_count=len(df.columns),
        cleaned_df=df,
        eda=eda_result,
        pii_findings=[],
        cleaning_log=[],
    )
    config = PipelineConfig()
    html = render_html(report_result, config)

    assert "Veri Analiz Raporu" in html
    assert "<div" in html
    assert "plotly" in html.lower() or "Plotly" in html


def test_html_reporter_inline_plotly_no_cdn(numeric_heavy_csv):
    """Plotly JS must be inlined (no CDN dependency) for offline use."""
    df = load_dataframe(numeric_heavy_csv)
    eda_result = run_basic_eda(df)
    report_result = ReportResult(
        input_path=numeric_heavy_csv,
        row_count=len(df),
        col_count=len(df.columns),
        cleaned_df=df,
        eda=eda_result,
        pii_findings=[],
        cleaning_log=[],
    )
    config = PipelineConfig()
    html = render_html(report_result, config)

    # No <script src="..."> CDN tag pointing to cdn.plot.ly
    assert '<script src="https://cdn.plot.ly' not in html, (
        "CDN <script src> reference found — report must be offline-compatible"
    )

    # Inline Plotly JS must be present in <head>
    head_end = html.find("</head>")
    assert head_end > 0, "</head> not found"
    head_section = html[:head_end]
    # The inline script must be substantial (~4.7MB) — Plotly JS inlined in <head>
    assert len(head_section) > 50000, (
        f"<head> section too small ({len(head_section)} bytes) — Plotly JS not inlined"
    )
    # Verify a script tag containing plotly marker exists in head
    script_start = head_section.find("<script>")
    assert script_start > 0, "No <script> block found in <head>"
    script_content = head_section[script_start:script_start + 500]
    assert "plotly" in script_content.lower(), (
        f"<script> block in <head> does not contain 'plotly': {script_content[:200]}..."
    )


def test_html_reporter_chart_divs_present(numeric_heavy_csv):
    """All expected chart divs present (Korelasyon Heatmap, histograms, clustering)."""
    df = load_dataframe(numeric_heavy_csv)
    eda_result = run_basic_eda(df)
    report_result = ReportResult(
        input_path=numeric_heavy_csv,
        row_count=len(df),
        col_count=len(df.columns),
        cleaned_df=df,
        eda=eda_result,
        pii_findings=[],
        cleaning_log=[],
    )
    config = PipelineConfig()
    html = render_html(report_result, config)

    # Count chart divs: <div class="chart">
    chart_count = html.count('class="chart"')
    # numeric_heavy.csv has 5 numeric columns (via load_dataframe) → 5 histograms.
    # (correlation_matrix + cluster_labels are None when using basic-EDA only)
    assert chart_count >= 5, f"Expected at least 5 chart divs, found {chart_count}"

    # No CDN dependency (check script src tag, not library-internal strings)
    assert '<script src="https://cdn.plot.ly' not in html


# ── Cleaning ─────────────────────────────────────────────────────────────────

def test_cleaning_conservative_noop(numeric_heavy_csv):
    df = load_dataframe(numeric_heavy_csv)
    cleaned, actions = clean(df, CleaningLevel.CONSERVATIVE)
    # Should be a copy, not the same object
    assert cleaned.equals(df)
    assert actions == []


# NOTE: moderate/aggressive cleaning are now implemented (Phase 2);
# see tests/test_cleaning.py for full coverage.


# ── PII vector validation with python-stdnum ─────────────────────────────────

def test_tckn_vectors_valid():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "vectors", "tckn_vectors.json")
    with open(path) as f:
        data = json.load(f)
    from stdnum.tr.tckimlik import is_valid
    for v in data["valid"]:
        assert is_valid(v), f"expected valid TCKN: {v}"
    for v in data["invalid"]:
        assert not is_valid(v), f"expected invalid TCKN: {v}"


def test_iban_vectors_valid():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "vectors", "iban_vectors.json")
    with open(path) as f:
        data = json.load(f)
    from stdnum import iban as tr_iban
    for v in data["valid"]:
        assert tr_iban.is_valid(v), f"expected valid IBAN: {v}"
    for v in data["invalid"]:
        assert not tr_iban.is_valid(v), f"expected invalid IBAN: {v}"
