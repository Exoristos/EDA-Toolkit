"""Microsoft Word (.docx) bank report renderer for eda_toolkit.

Produces a structured Turkish-language data analysis report with embedded
PNG charts (via viz_static). Defensive: skips sections whose source data
is None or empty.
"""

from __future__ import annotations

import io
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

from eda_toolkit.types import ReportResult, PipelineConfig

# ── constants ────────────────────────────────────────────────────────────────

HEADING_SIZE = Pt(14)
SUBHEADING_SIZE = Pt(12)
BODY_SIZE = Pt(9)
HEADER_BG = "1E3A5F"
BORDER_COLOR = "D1D5DB"

MAX_HISTOGRAMS = 20  # cap on distribution plots per report


# ── helpers ──────────────────────────────────────────────────────────────────


def _set_cell_shading(cell, color: str) -> None:
    """Set background fill for a table cell."""
    shading_elm = cell._element.get_or_add_tcPr()
    shading = shading_elm.makeelement(qn("w:shd"), {
        qn("w:fill"): color,
        qn("w:val"): "clear",
    })
    shading_elm.append(shading)


def _add_styled_table(doc, headers: list[str], rows: list[list], col_widths: list[Inches] | None = None):
    """Add a table with styled header row (dark bg, white text) and alternating body rows."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"

    # Header
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = str(h)
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
                run.font.size = BODY_SIZE
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        _set_cell_shading(cell, HEADER_BG)

    # Data
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = str(val) if val is not None else ""
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = BODY_SIZE
            if i % 2 == 0:
                _set_cell_shading(cell, "F3F4F6")

    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = w

    return table


def _add_heading(doc, text: str, level: int = 1) -> None:
    """Add a styled heading paragraph."""
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x5F)


def _safe_df(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return df if non-empty, else empty DataFrame."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return pd.DataFrame()
    return df


# ── main renderer ────────────────────────────────────────────────────────────


def render_docx(result: ReportResult, config: PipelineConfig, out_path: str) -> None:
    """Render a multi-section Word report and write to out_path.

    Sections:
      - Baslik (title + file info)
      - Ozet (row/col counts, schema table)
      - Tanimlayici Istatistik
      - Eksik Veri (table + missing bar chart)
      - Korelasyon (correlation heatmap)
      - Dagilimlar (histogram per numeric column, capped)
      - Aykiri Degerler
      - Hipotez Testleri
      - PII Uyarilari
      - Temizleme Islemleri

    Every section is skipped when its source data is None or empty.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Lazy imports to keep viz_static optional at module level
    from eda_toolkit.viz_static import (
        png_correlation_heatmap,
        png_histogram,
        png_missing_bar,
    )

    doc = Document()

    # Set default font
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Calibri"
    font.size = BODY_SIZE

    # ── 1. Baslik ─────────────────────────────────────────────────────────
    title = doc.add_heading("Veri Analiz Raporu", level=0)
    for run in title.runs:
        run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x5F)

    input_name = Path(result.input_path).name
    date_str = datetime.now().strftime("%d.%m.%Y %H:%M")
    p = doc.add_paragraph(f"Dosya: {input_name}  |  Olusturma: {date_str}")
    p.runs[0].font.size = Pt(10)

    doc.add_paragraph()  # spacer

    # ── 2. Ozet ───────────────────────────────────────────────────────────
    _add_heading(doc, "Ozet Bilgiler", 1)

    # cleaning summary
    total_actions = len(result.cleaning_log)
    applied_actions = sum(1 for a in result.cleaning_log if a.applied)

    # PII summary
    pii_counts: dict[str, int] = {}
    for f in result.pii_findings:
        pii_counts[f.pii_type.value] = pii_counts.get(f.pii_type.value, 0) + 1
    pii_summary_str = ", ".join(f"{k}: {v}" for k, v in pii_counts.items()) if pii_counts else "yok"

    summary_rows = [
        ["Dosya", result.input_path],
        ["Satir Sayisi", str(result.row_count)],
        ["Sutun Sayisi", str(result.col_count)],
        ["Temizleme Seviyesi", config.cleaning.value],
        ["Temizleme Aksiyon Sayisi", str(total_actions)],
        ["Uygulanan Aksiyon", str(applied_actions)],
        ["PII Bulgulari (tur:adet)", pii_summary_str],
    ]
    _add_styled_table(doc, ["Metrik", "Deger"], summary_rows,
                      col_widths=[Inches(2.5), Inches(4.0)])

    # Schema table
    if result.eda.schemas:
        _add_heading(doc, "Sutun Semasi", 2)
        schema_rows = [
            [s.name, s.dtype, s.kind, str(s.unique_count), str(s.missing_count)]
            for s in result.eda.schemas
        ]
        _add_styled_table(doc, ["Sutun", "Veri Tipi", "Tur", "Eslesik Sayisi", "Eksik Sayisi"],
                          schema_rows)

    doc.add_page_break()

    # ── 3. Tanimlayici Istatistik ─────────────────────────────────────────
    desc = result.eda.descriptive
    if not desc.empty:
        _add_heading(doc, "Tanimlayici Istatistik", 1)
        headers = ["Istatistik"] + list(desc.columns)
        rows = []
        for stat_name, row_vals in desc.iterrows():
            rows.append([str(stat_name)] + [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row_vals])
        _add_styled_table(doc, headers, rows)
        doc.add_page_break()

    # ── 4. Eksik Veri ─────────────────────────────────────────────────────
    missing_df = result.eda.missing_summary
    if not missing_df.empty:
        _add_heading(doc, "Eksik Veri Ozeti", 1)
        _add_heading(doc, "Eksik Veri Tablosu", 2)
        m_headers = list(missing_df.columns)
        m_rows = [[str(v) for v in row] for _, row in missing_df.iterrows()]
        _add_styled_table(doc, m_headers, m_rows)

        # Missing bar chart
        png_data = png_missing_bar(missing_df)
        if png_data is not None:
            doc.add_paragraph()
            _add_heading(doc, "Eksik Veri Grafigi", 2)
            doc.add_picture(io.BytesIO(png_data), width=Inches(5.5))

        doc.add_page_break()

    # ── 5. Korelasyon ─────────────────────────────────────────────────────
    corr = result.eda.correlation_matrix
    if corr is not None and not corr.empty and corr.shape[0] >= 2:
        _add_heading(doc, "Korelasyon Matrisi", 1)
        png_data = png_correlation_heatmap(corr)
        if png_data is not None:
            doc.add_picture(io.BytesIO(png_data), width=Inches(6.0))
        doc.add_page_break()

    # ── 6. Dagilimlar (Histograms) ────────────────────────────────────────
    cleaned = result.cleaned_df
    if cleaned is not None and not cleaned.empty:
        numeric_cols = cleaned.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            _add_heading(doc, "Dagilimlar", 1)
            count = 0
            for col in numeric_cols[:MAX_HISTOGRAMS]:
                series = cleaned[col]
                png_data = png_histogram(series, col)
                if png_data is not None:
                    _add_heading(doc, col, 2)
                    doc.add_picture(io.BytesIO(png_data), width=Inches(5.5))
                    count += 1
            if count > 0:
                doc.add_page_break()

    # ── 7. Aykiri Degerler ────────────────────────────────────────────────
    outliers = result.eda.outliers
    if outliers is not None and len(outliers) > 0:
        _add_heading(doc, "Aykiri Degerler", 1)
        out_rows = []
        for col_name, info in outliers.items():
            cnt = info.get("count", 0)
            rate = info.get("rate", 0.0)
            lower = info.get("lower")
            upper = info.get("upper")
            out_rows.append([
                col_name,
                str(cnt),
                f"{rate:.4f}",
                f"{lower:.4f}" if lower is not None else "-",
                f"{upper:.4f}" if upper is not None else "-",
            ])
        _add_styled_table(doc, ["Sutun", "Adet", "Oran", "Alt Sinir", "Ust Sinir"], out_rows)
        doc.add_page_break()

    # ── 8. Hipotez Testleri ───────────────────────────────────────────────
    hyp_tests = result.eda.hypothesis_tests
    if hyp_tests is not None and len(hyp_tests) > 0:
        _add_heading(doc, "Hipotez Testleri", 1)
        h_rows = []
        for t in hyp_tests:
            test_type = t.get("test", "")
            if test_type == "shapiro":
                test_label = "Shapiro-Wilk"
                col_str = str(t.get("column", ""))
                is_normal = t.get("is_normal")
                sonuc = "Normal" if is_normal is True else ("Normal degil" if is_normal is False else "Belirsiz")
            elif test_type == "pearson":
                test_label = "Pearson"
                cols = t.get("columns", [])
                col_str = ", ".join(cols) if cols else ""
                is_significant = t.get("significant")
                sonuc = "Anlamli" if is_significant is True else ("Anlamsiz" if is_significant is False else "Belirsiz")
            else:
                test_label = test_type
                col_str = str(t.get("column", t.get("columns", "")))
                sonuc = ""
            stat = t.get("statistic")
            stat_str = f"{stat:.4f}" if stat is not None else "-"
            p_val = t.get("p_value")
            p_str = f"{p_val:.4f}" if p_val is not None else "-"
            h_rows.append([test_label, col_str, stat_str, p_str, sonuc])
        _add_styled_table(doc, ["Tur", "Sutun", "Istatistik", "p-degeri", "Sonuc"], h_rows)
        doc.add_page_break()

    # ── 9. PII Uyarilari ──────────────────────────────────────────────────
    _add_heading(doc, "PII Uyarilari", 1)
    if result.pii_findings:
        pii_rows = [
            [
                f.column,
                f.pii_type.value,
                f"{f.confidence * 100:.1f}",
                f"{f.match_rate * 100:.1f}",
                str(f.sample_count),
                f.detection_layer,
            ]
            for f in result.pii_findings
        ]
        _add_styled_table(doc, ["Sutun", "PII Turu", "Guven %", "Eslesme Orani %", "Ornek Sayisi", "Tespit Katmani"],
                          pii_rows)
    else:
        doc.add_paragraph("Kisisel veri (PII) tespit edilmedi.")
    doc.add_page_break()

    # ── 10. Temizleme Islemleri ────────────────────────────────────────────
    _add_heading(doc, "Temizleme Islemleri", 1)
    if result.cleaning_log:
        clean_rows = [
            [a.column, a.issue, a.level.value, "Evet" if a.applied else "Hayir", a.detail]
            for a in result.cleaning_log
        ]
        _add_styled_table(doc, ["Sutun", "Sorun", "Seviye", "Uygulandi", "Detay"], clean_rows)
    else:
        doc.add_paragraph("Temizleme islemi uygulanmadi.")

    # ── Save ──────────────────────────────────────────────────────────────
    doc.save(out_path)
