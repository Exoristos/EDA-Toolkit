from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

from eda_toolkit.types import ReportResult, PipelineConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEADER_BG = colors.Color(0.118, 0.227, 0.373)  # #1e3a5f
HEADER_FG = colors.white
ALT_ROW_BG = colors.Color(0.953, 0.957, 0.965)  # #f3f4f6
GRID_COLOR = colors.Color(0.820, 0.835, 0.859)  # #d1d5db

PAGE_W, PAGE_H = landscape(A4)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

_BASE_STYLES = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle(
    "EDATitle",
    parent=_BASE_STYLES["Heading1"],
    fontSize=18,
    textColor=HEADER_BG,
    spaceAfter=12,
    leading=22,
)

H2_STYLE = ParagraphStyle(
    "EDAH2",
    parent=_BASE_STYLES["Heading2"],
    fontSize=13,
    textColor=colors.Color(0.173, 0.322, 0.510),  # #2c5282
    spaceBefore=16,
    spaceAfter=8,
    leading=16,
)

BODY_STYLE = ParagraphStyle(
    "EDABody",
    parent=_BASE_STYLES["Normal"],
    fontSize=9,
    leading=12,
    spaceAfter=4,
)

CELL_STYLE = ParagraphStyle(
    "EDACell",
    parent=BODY_STYLE,
    fontSize=8,
    leading=10,
)

HEADER_CELL_STYLE = ParagraphStyle(
    "EDAHeaderCell",
    parent=CELL_STYLE,
    textColor=HEADER_FG,
    fontName="Helvetica-Bold",
    fontSize=8,
    leading=10,
)

TABLE_STYLE_BASE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
    ("TEXTCOLOR", (0, 0), (-1, 0), HEADER_FG),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("GRID", (0, 0), (-1, -1), 0.5, GRID_COLOR),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
])

# ---------------------------------------------------------------------------
# Footer callback
# ---------------------------------------------------------------------------


def _on_page(canvas, doc):
    """Add page number to bottom-right of each page."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.Color(0.420, 0.447, 0.502))  # #6b7280
    page_text = f"Sayfa {canvas.getPageNumber()}"
    canvas.drawRightString(PAGE_W - 2 * cm, 1.5 * cm, page_text)
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def _p(text: str, style: ParagraphStyle = BODY_STYLE) -> Paragraph:
    """Make a Paragraph, escaping XML-special characters."""
    from xml.sax.saxutils import escape
    # Don't double-escape ampersands that are already part of entities
    safe = escape(text, {'"': "&quot;"})
    return Paragraph(safe, style)


def _header_row(labels: list[str]) -> list[Paragraph]:
    from xml.sax.saxutils import escape
    return [Paragraph(escape(str(lbl), {'"': "&quot;"}), HEADER_CELL_STYLE) for lbl in labels]


def _data_row(values: list, style: ParagraphStyle = CELL_STYLE) -> list[Paragraph]:
    from xml.sax.saxutils import escape
    return [Paragraph(escape(str(v), {'"': "&quot;"}), style) for v in values]


def _df_to_table(df: pd.DataFrame, max_rows: int = 50, round_decimals: int | None = 2) -> tuple:
    """Convert a DataFrame to a Platypus Table, plus note if truncated.

    Returns (elements, str_note_or_none).
    """
    if df.empty:
        return ([_p("Veri yok.")], None)

    original_len = len(df)
    truncated = False
    if original_len > max_rows:
        df = df.head(max_rows)
        truncated = True

    # Build header + data
    header = _header_row([str(c) for c in df.columns])
    data = [header]
    for _, row in df.iterrows():
        vals = []
        for v in row:
            if isinstance(v, float):
                if round_decimals is not None:
                    vals.append(f"{v:.{round_decimals}f}")
                else:
                    vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        data.append(_data_row(vals))

    t = Table(data)
    style_cmds = list(TABLE_STYLE_BASE.getCommands())
    # Alternating row background for data rows
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), ALT_ROW_BG))
    t.setStyle(TableStyle(style_cmds))

    note = None
    if truncated:
        note = f"(Tablo ilk {max_rows} satiri gostermektedir. Toplam: {original_len} satir.)"
    return (t, note)


def _dict_list_to_table(
    rows: list[dict],
    columns: list[str],
    header_labels: list[str] | None = None,
) -> Table:
    """Convert a list of dicts to a Platypus Table."""
    if not rows:
        return Table([["Veri yok."]])

    labels = header_labels if header_labels else columns
    header = _header_row(labels)
    table_data = [header]
    for row in rows:
        vals = [str(row.get(col, "")) for col in columns]
        table_data.append(_data_row(vals))

    t = Table(table_data)
    style_cmds = list(TABLE_STYLE_BASE.getCommands())
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), ALT_ROW_BG))
    t.setStyle(TableStyle(style_cmds))
    return t


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------


def render_pdf(result: ReportResult, config: PipelineConfig, out_path: str) -> str:
    """Render a multi-section PDF report and write to out_path.

    Sections (skipped when source data is missing/empty):
      1. Baslik (Title) — "Veri Analiz Raporu"
      2. Ozet (Summary) — row/col counts, cleaning summary, PII summary
      3. Eksik Veri Ozeti (Missing) — missing_summary table
      4. PII Uyarilari (PII alerts) — pii_findings table
      5. Korelasyon (optional) — correlation_matrix table
      6. Aykiri Degerler (optional) — outliers table
      7. Hipotez Testleri (optional) — hypothesis_tests table

    Returns out_path.
    Creates the parent directory when it does not exist.
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    doc = SimpleDocTemplate(
        out_path,
        pagesize=(PAGE_W, PAGE_H),
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=2.0 * cm,
        title="Veri Analiz Raporu",
        author="eda_toolkit",
    )

    story: list = []

    # -----------------------------------------------------------------------
    # 1. Baslik (Title)
    # -----------------------------------------------------------------------
    story.append(Paragraph("Veri Analiz Raporu", TITLE_STYLE))
    input_name = Path(result.input_path).name
    date_str = datetime.now().strftime("%d.%m.%Y %H:%M")
    story.append(_p(f"Dosya: {input_name}  |  Olusturma: {date_str}"))
    story.append(Spacer(1, 0.5 * cm))

    # -----------------------------------------------------------------------
    # 2. Ozet (Summary)
    # -----------------------------------------------------------------------
    story.append(Paragraph("Ozet Bilgiler", H2_STYLE))

    # Cleaning summary
    total_actions = len(result.cleaning_log)
    applied_actions = sum(1 for a in result.cleaning_log if a.applied)

    # PII summary
    pii_counts: dict[str, int] = {}
    for f in result.pii_findings:
        pii_counts[f.pii_type.value] = pii_counts.get(f.pii_type.value, 0) + 1
    pii_summary_str = ", ".join(f"{k}: {v}" for k, v in pii_counts.items()) if pii_counts else "yok"

    summary_data = [
        _header_row(["Metrik", "Deger"]),
        _data_row(["Dosya", result.input_path]),
        _data_row(["Satir Sayisi", str(result.row_count)]),
        _data_row(["Sutun Sayisi", str(result.col_count)]),
        _data_row(["Temizleme Seviyesi", config.cleaning.value]),
        _data_row(["Temizleme Aksiyon Sayisi", str(total_actions)]),
        _data_row(["Uygulanan Aksiyon", str(applied_actions)]),
        _data_row(["PII Bulgulari (tur:adet)", pii_summary_str]),
    ]
    summary_table = Table(summary_data, colWidths=[6 * cm, 16 * cm])
    style_cmds = list(TABLE_STYLE_BASE.getCommands())
    for i in range(1, len(summary_data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), ALT_ROW_BG))
    summary_table.setStyle(TableStyle(style_cmds))
    story.append(summary_table)
    story.append(Spacer(1, 0.4 * cm))

    # -----------------------------------------------------------------------
    # 3. Eksik Veri Ozeti (Missing)
    # -----------------------------------------------------------------------
    story.append(Paragraph("Eksik Veri Ozeti", H2_STYLE))
    missing_df: pd.DataFrame = result.eda.missing_summary
    if not missing_df.empty:
        table, note = _df_to_table(missing_df, max_rows=50, round_decimals=4)
        story.append(table)
        if note:
            story.append(Spacer(1, 0.2 * cm))
            story.append(_p(note))
    else:
        story.append(_p("Eksik veri bulunamadi."))
    story.append(Spacer(1, 0.4 * cm))

    # -----------------------------------------------------------------------
    # 4. PII Uyarilari (PII alerts)
    # -----------------------------------------------------------------------
    story.append(Paragraph("PII Uyarilari", H2_STYLE))
    if result.pii_findings:
        pii_rows = [
            {
                "Sutun": f.column,
                "Tur": f.pii_type.value,
                "Guven %": f"{f.confidence * 100:.1f}",
                "Eslesme Orani %": f"{f.match_rate * 100:.1f}",
                "Katman": f.detection_layer,
            }
            for f in result.pii_findings
        ]
        pii_table = _dict_list_to_table(
            pii_rows,
            columns=["Sutun", "Tur", "Guven %", "Eslesme Orani %", "Katman"],
        )
        story.append(pii_table)
    else:
        story.append(_p("Kisisel veri (PII) tespit edilmedi."))
    story.append(Spacer(1, 0.4 * cm))

    # -----------------------------------------------------------------------
    # 5. Korelasyon (optional)
    # -----------------------------------------------------------------------
    corr = result.eda.correlation_matrix
    if corr is not None and not corr.empty and corr.shape[0] >= 2:
        story.append(Paragraph("Korelasyon Matrisi", H2_STYLE))
        table, _ = _df_to_table(corr, max_rows=50, round_decimals=2)
        story.append(table)
        story.append(Spacer(1, 0.4 * cm))

    # -----------------------------------------------------------------------
    # 6. Aykiri Degerler (optional)
    # -----------------------------------------------------------------------
    outliers = result.eda.outliers
    if outliers is not None and len(outliers) > 0:
        story.append(Paragraph("Aykiri Degerler", H2_STYLE))
        out_rows = []
        for col_name, info in outliers.items():
            cnt = info.get("count", 0)
            rate = info.get("rate", 0.0)
            lower = info.get("lower")
            upper = info.get("upper")
            out_rows.append({
                "Sutun": col_name,
                "Adet": str(cnt),
                "Oran": f"{rate:.4f}",
                "Alt Sinir": f"{lower:.4f}" if lower is not None else "-",
                "Ust Sinir": f"{upper:.4f}" if upper is not None else "-",
            })
        out_table = _dict_list_to_table(
            out_rows,
            columns=["Sutun", "Adet", "Oran", "Alt Sinir", "Ust Sinir"],
        )
        story.append(out_table)
        story.append(Spacer(1, 0.4 * cm))

    # -----------------------------------------------------------------------
    # 7. Hipotez Testleri (optional)
    # -----------------------------------------------------------------------
    hyp_tests = result.eda.hypothesis_tests
    if hyp_tests is not None and len(hyp_tests) > 0:
        story.append(Paragraph("Hipotez Testleri", H2_STYLE))
        h_rows = []
        for t in hyp_tests:
            test_type = t.get("test", "")
            if test_type == "shapiro":
                test_label = "Shapiro-Wilk"
                col_str = str(t.get("column", ""))
                is_normal = t.get("is_normal")
                if is_normal is True:
                    sonuc = "Normal"
                elif is_normal is False:
                    sonuc = "Normal degil"
                else:
                    sonuc = "Belirsiz"
            elif test_type == "pearson":
                test_label = "Pearson"
                cols = t.get("columns", [])
                col_str = ", ".join(cols) if cols else ""
                is_significant = t.get("significant")
                if is_significant is True:
                    sonuc = "Anlamli"
                elif is_significant is False:
                    sonuc = "Anlamsiz"
                else:
                    sonuc = "Belirsiz"
            else:
                test_label = test_type
                col_str = str(t.get("column", t.get("columns", "")))
                sonuc = ""

            stat = t.get("statistic")
            stat_str = f"{stat:.4f}" if stat is not None else "-"
            p_val = t.get("p_value")
            p_str = f"{p_val:.4f}" if p_val is not None else "-"

            h_rows.append({
                "Tur": test_label,
                "Sutun": col_str,
                "Istatistik": stat_str,
                "p-degeri": p_str,
                "Sonuc": sonuc,
            })
        hyp_table = _dict_list_to_table(
            h_rows,
            columns=["Tur", "Sutun", "Istatistik", "p-degeri", "Sonuc"],
        )
        story.append(hyp_table)
        story.append(Spacer(1, 0.4 * cm))

    # Build the PDF
    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)

    return out_path


def write_pdf(result: ReportResult, config: PipelineConfig, out_path: str) -> str:
    """Thin alias for render_pdf (mirrors html_reporter / excel_reporter pattern)."""
    return render_pdf(result, config, out_path)
