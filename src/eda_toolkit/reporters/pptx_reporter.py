"""Microsoft PowerPoint (.pptx) bank report renderer for eda_toolkit.

Produces a Turkish-language deck with embedded PNG charts (via viz_static).
Supports optional .potx template. Defensive: skips slides when source data
is None or empty.
"""

from __future__ import annotations

import io
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

from eda_toolkit.types import ReportResult, PipelineConfig

# ── constants ────────────────────────────────────────────────────────────────

HEADER_BG_HEX = "1E3A5F"
HEADER_RGB = RGBColor(0x1E, 0x3A, 0x5F)
BODY_RGB = RGBColor(0x33, 0x33, 0x33)
WHITE_RGB = RGBColor(0xFF, 0xFF, 0xFF)

MAX_HISTOGRAMS = 20

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# Image placement regions
IMG_LEFT = Inches(0.8)
IMG_TOP = Inches(1.6)
IMG_W = Inches(7.5)
IMG_H = Inches(5.0)

TABLE_LEFT = Inches(0.5)
TABLE_TOP = Inches(1.5)
TABLE_W = Inches(12.3)


# ── helpers ──────────────────────────────────────────────────────────────────


def _add_title(slide, text: str, top: Inches = Inches(0.3), left: Inches = Inches(0.5),
               width: Inches = Inches(12.3), height: Inches = Inches(0.8)) -> None:
    """Add a styled title text box to a slide."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = HEADER_RGB


def _add_body(slide, text: str, top: Inches, left: Inches = Inches(0.5),
              width: Inches = Inches(12.3), size: Pt = Pt(14)) -> None:
    """Add a body text box."""
    txBox = slide.shapes.add_textbox(left, top, width, Inches(0.5))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = size
    p.font.color.rgb = BODY_RGB


def _add_bullets(slide, lines: list[str], top: Inches, left: Inches = Inches(0.8),
                 width: Inches = Inches(11.5), size: Pt = Pt(13)) -> None:
    """Add a bulleted text box with multiple lines."""
    txBox = slide.shapes.add_textbox(left, top, width, Inches(2.5))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = size
        p.font.color.rgb = BODY_RGB
        p.space_after = Pt(4)


def _add_image(slide, png_bytes: bytes, left: Inches = IMG_LEFT, top: Inches = IMG_TOP,
               width: Inches = IMG_W, height: Inches = IMG_H) -> None:
    """Embed a PNG image on a slide."""
    stream = io.BytesIO(png_bytes)
    slide.shapes.add_picture(stream, left, top, width, height)


def _add_table(slide, headers: list[str], rows: list[list],
               left: Inches = TABLE_LEFT, top: Inches = TABLE_TOP,
               width: Inches = TABLE_W) -> None:
    """Add a styled table to a slide."""
    num_rows = 1 + len(rows)
    num_cols = len(headers)
    table_shape = slide.shapes.add_table(num_rows, num_cols, left, top, width, Inches(0.4 * num_rows))
    table = table_shape.table

    # Header row
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = str(h)
        for paragraph in cell.text_frame.paragraphs:
            paragraph.font.size = Pt(10)
            paragraph.font.bold = True
            paragraph.font.color.rgb = WHITE_RGB
            paragraph.alignment = PP_ALIGN.CENTER
        _cell_fill(cell, HEADER_BG_HEX)

    # Data rows
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = str(val) if val is not None else ""
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(9)
                paragraph.font.color.rgb = BODY_RGB
            if i % 2 == 0:
                _cell_fill(cell, "F3F4F6")


def _cell_fill(cell, color_hex: str) -> None:
    """Set solid fill for a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    solidFill = tcPr.makeelement(
        "{http://schemas.openxmlformats.org/drawingml/2006/main}solidFill",
        {},
    )
    srgbClr = solidFill.makeelement(
        "{http://schemas.openxmlformats.org/drawingml/2006/main}srgbClr",
        {"val": color_hex},
    )
    solidFill.append(srgbClr)
    tcPr.append(solidFill)


def _add_page_number(slide, num: int, total: int) -> None:
    """Add a small page number in the bottom-right corner."""
    txBox = slide.shapes.add_textbox(Inches(12.0), Inches(7.0), Inches(1.0), Inches(0.3))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = f"{num}/{total}"
    p.font.size = Pt(8)
    p.font.color.rgb = RGBColor(0x9C, 0xA3, 0xAF)
    p.alignment = PP_ALIGN.RIGHT


# ── main renderer ────────────────────────────────────────────────────────────


def render_pptx(result: ReportResult, config: PipelineConfig, out_path: str,
                template: str | None = None) -> None:
    """Render a multi-slide PowerPoint report and write to out_path.

    Slides:
      - Kapak (cover: title, file name, date)
      - Ozet (row/col counts, schema stats)
      - Korelasyon (correlation heatmap + bullets)
      - Dagilimlar (one slide per histogram, capped)
      - Aykiri + Hipotez (combined slide with tables)
      - PII Uyarilari (table + risk note)
      - Temizleme Islemleri (table)

    Optionally loads a .potx template via the *template* parameter.
    Every slide is skipped when its source data is None or empty.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Lazy imports to keep viz_static optional at module level
    from eda_toolkit.viz_static import (
        png_correlation_heatmap,
        png_histogram,
        png_missing_bar,
    )

    # Load presentation (template or default 16:9)
    if template and os.path.isfile(str(template)):
        prs = Presentation(str(template))
    else:
        prs = Presentation()
        prs.slide_width = SLIDE_W
        prs.slide_height = SLIDE_H

    slides = []  # accumulate slides for page numbering

    # ── 1. Kapak ──────────────────────────────────────────────────────────
    slide_layout = prs.slide_layouts[0] if template and os.path.isfile(str(template)) else prs.slide_layouts[6]  # blank
    slide = prs.slides.add_slide(slide_layout)
    slides.append(slide)

    _add_title(slide, "Veri Analiz Raporu", top=Inches(2.0))
    input_name = Path(result.input_path).name
    date_str = datetime.now().strftime("%d.%m.%Y %H:%M")
    _add_body(slide, f"Dosya: {input_name}", top=Inches(3.2), size=Pt(16))
    _add_body(slide, f"Olusturma: {date_str}", top=Inches(3.8), size=Pt(13))

    # ── 2. Ozet ───────────────────────────────────────────────────────────
    slide = prs.slides.add_slide(_blank_layout(prs))
    slides.append(slide)

    _add_title(slide, "Ozet Bilgiler")

    total_actions = len(result.cleaning_log)
    applied_actions = sum(1 for a in result.cleaning_log if a.applied)

    pii_counts: dict[str, int] = {}
    for f in result.pii_findings:
        pii_counts[f.pii_type.value] = pii_counts.get(f.pii_type.value, 0) + 1
    pii_summary_str = ", ".join(f"{k}: {v}" for k, v in pii_counts.items()) if pii_counts else "yok"

    summary_rows = [
        ["Dosya", result.input_path],
        ["Satir Sayisi", str(result.row_count)],
        ["Sutun Sayisi", str(result.col_count)],
        ["Temizleme Seviyesi", config.cleaning.value],
        ["Aksiyon / Uygulanan", f"{total_actions} / {applied_actions}"],
        ["PII Bulgulari", pii_summary_str],
    ]
    _add_table(slide, ["Metrik", "Deger"], summary_rows, top=Inches(1.3))

    # Optional PII note
    if result.pii_findings:
        _add_body(slide, f"Toplam {len(result.pii_findings)} PII bulgusu tespit edildi.", top=Inches(5.5), size=Pt(12))

    # ── 3. Korelasyon ─────────────────────────────────────────────────────
    corr = result.eda.correlation_matrix
    if corr is not None and not corr.empty and corr.shape[0] >= 2:
        slide = prs.slides.add_slide(_blank_layout(prs))
        slides.append(slide)
        _add_title(slide, "Korelasyon Matrisi")

        png_data = png_correlation_heatmap(corr)
        if png_data is not None:
            _add_image(slide, png_data, left=Inches(0.5), top=Inches(1.3),
                       width=Inches(6.5), height=Inches(5.5))

        # Bullet points
        bullets = ["Korelasyon matrisi Pearson katsayilarini gostermektedir.",
                   "Pozitif degerler dogrusal pozitif iliskiyi, negatif degerler ters iliskiyi ifade eder.",
                   "1.0 ve -1.0 degerleri mukemmel dogrusal iliskiyi gosterir."]
        _add_bullets(slide, bullets, top=Inches(1.4), left=Inches(7.5), width=Inches(5.0))

    # ── 4. Dagilimlar (Histograms) ────────────────────────────────────────
    cleaned = result.cleaned_df
    if cleaned is not None and not cleaned.empty:
        numeric_cols = cleaned.select_dtypes(include=["number"]).columns.tolist()
        for col in numeric_cols[:MAX_HISTOGRAMS]:
            series = cleaned[col]
            png_data = png_histogram(series, col)
            if png_data is not None:
                slide = prs.slides.add_slide(_blank_layout(prs))
                slides.append(slide)
                _add_title(slide, f"Dagilim: {col}")
                _add_image(slide, png_data, left=Inches(1.0), top=Inches(1.3),
                           width=Inches(11.0), height=Inches(5.5))

    # ── 5. Aykiri + Hipotez combined ──────────────────────────────────────
    outliers = result.eda.outliers
    hyp_tests = result.eda.hypothesis_tests
    if (outliers is not None and len(outliers) > 0) or (hyp_tests is not None and len(hyp_tests) > 0):
        slide = prs.slides.add_slide(_blank_layout(prs))
        slides.append(slide)
        _add_title(slide, "Aykiri Degerler ve Hipotez Testleri")

        current_top = Inches(1.3)

        # Outliers
        if outliers is not None and len(outliers) > 0:
            _add_body(slide, "Aykiri Degerler:", top=current_top, size=Pt(14))
            current_top += Inches(0.35)
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
            num_out_rows = len(out_rows)
            _add_table(slide, ["Sutun", "Adet", "Oran", "Alt Sinir", "Ust Sinir"], out_rows,
                       top=current_top, width=Inches(11.0))
            current_top += Inches(0.5 * (num_out_rows + 2))

        # Hypothesis tests
        if hyp_tests is not None and len(hyp_tests) > 0:
            _add_body(slide, "Hipotez Testleri:", top=current_top, size=Pt(14))
            current_top += Inches(0.35)
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
            num_h_rows = len(h_rows)
            _add_table(slide, ["Tur", "Sutun", "Istatistik", "p-degeri", "Sonuc"], h_rows,
                       top=current_top, width=Inches(11.0))

    # ── 6. PII Uyarilari ──────────────────────────────────────────────────
    slide = prs.slides.add_slide(_blank_layout(prs))
    slides.append(slide)
    _add_title(slide, "PII Uyarilari")

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
        _add_table(slide, ["Sutun", "PII Turu", "Guven %", "Eslesme Orani %", "Ornek Sayisi", "Tespit Katmani"],
                   pii_rows, top=Inches(1.3))
        _add_body(slide, "Not: Kisisel veri iceren alanlar raporlanmistir. Veri guvenligi politikalarina uygun hareket edilmelidir.",
                  top=Inches(6.0), size=Pt(11))
    else:
        _add_body(slide, "Kisisel veri (PII) tespit edilmedi.", top=Inches(2.0), size=Pt(16))

    # ── 7. Temizleme Islemleri ────────────────────────────────────────────
    slide = prs.slides.add_slide(_blank_layout(prs))
    slides.append(slide)
    _add_title(slide, "Temizleme Islemleri")

    if result.cleaning_log:
        clean_rows = [
            [a.column, a.issue, a.level.value, "Evet" if a.applied else "Hayir", a.detail]
            for a in result.cleaning_log
        ]
        _add_table(slide, ["Sutun", "Sorun", "Seviye", "Uygulandi", "Detay"], clean_rows,
                   top=Inches(1.3))
    else:
        _add_body(slide, "Temizleme islemi uygulanmadi.", top=Inches(2.0), size=Pt(16))

    # ── Page numbers ──────────────────────────────────────────────────────
    total = len(slides)
    for i, slide in enumerate(slides, 1):
        _add_page_number(slide, i, total)

    # ── Save ──────────────────────────────────────────────────────────────
    prs.save(out_path)


def _blank_layout(prs: Presentation):
    """Return a blank layout, falling back to the first layout."""
    for layout in prs.slide_layouts:
        if layout.name and "blank" in layout.name.lower():
            return layout
    return prs.slide_layouts[6]  # typically blank
