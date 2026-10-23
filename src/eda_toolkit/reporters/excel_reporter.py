from __future__ import annotations

import os

import pandas as pd

from eda_toolkit.types import ReportResult, PipelineConfig


def render_excel(result: ReportResult, config: PipelineConfig, out_path: str) -> str:
    """Render a multi-sheet Excel report and write to out_path.

    Sheets produced (skipped when source data is missing/empty):
      Ozet              – row/col counts, input path, cleaning + PII summary
      Ham_Veri          – result.cleaned_df (raw data the report is based on)
      Tanimlayici_Istatistik – result.eda.descriptive
      Eksik_Veri        – result.eda.missing_summary
      Korelasyon        – result.eda.correlation_matrix (if not None)
      Temizleme_Logu    – flattened cleaning_log
      PII_Uyarilari     – flattened pii_findings

    Returns out_path.
    Creates the parent directory when it does not exist.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # ── Build summary row ─────────────────────────────────────────────────────
    # PII summary: count by type
    pii_counts: dict[str, int] = {}
    for f in result.pii_findings:
        pii_counts[f.pii_type.value] = pii_counts.get(f.pii_type.value, 0) + 1
    pii_summary_str = ", ".join(f"{k}: {v}" for k, v in pii_counts.items()) if pii_counts else "yok"

    # cleaning summary
    total_actions = len(result.cleaning_log)
    applied_actions = sum(1 for a in result.cleaning_log if a.applied)

    summary_data = {
        "Metrik": [
            "Dosya",
            "Satir Sayisi",
            "Sutun Sayisi",
            "Temizleme Aksiyon Sayisi",
            "Uygulanan Aksiyon Sayisi",
            "PII Bulgulari (tur:adet)",
        ],
        "Deger": [
            result.input_path,
            result.row_count,
            result.col_count,
            total_actions,
            applied_actions,
            pii_summary_str,
        ],
    }
    df_summary = pd.DataFrame(summary_data)

    # ── Write sheets ──────────────────────────────────────────────────────────
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # 1. Ozet (always written)
        df_summary.to_excel(writer, sheet_name="Ozet", index=False)

        # 2. Ham_Veri
        if result.cleaned_df is not None and not result.cleaned_df.empty:
            result.cleaned_df.to_excel(writer, sheet_name="Ham_Veri", index=False)
        else:
            pd.DataFrame({"bilgi": ["veri yok"]}).to_excel(
                writer, sheet_name="Ham_Veri", index=False
            )

        # 3. Tanimlayici_Istatistik
        if not result.eda.descriptive.empty:
            result.eda.descriptive.to_excel(
                writer, sheet_name="Tanimlayici_Istatistik", index=True
            )
        else:
            pd.DataFrame({"bilgi": ["veri yok"]}).to_excel(
                writer, sheet_name="Tanimlayici_Istatistik", index=False
            )

        # 4. Eksik_Veri
        if not result.eda.missing_summary.empty:
            result.eda.missing_summary.to_excel(
                writer, sheet_name="Eksik_Veri", index=False
            )
        else:
            pd.DataFrame({"bilgi": ["veri yok"]}).to_excel(
                writer, sheet_name="Eksik_Veri", index=False
            )

        # 5. Korelasyon — skipped when None/empty
        if result.eda.correlation_matrix is not None and not result.eda.correlation_matrix.empty:
            result.eda.correlation_matrix.to_excel(
                writer, sheet_name="Korelasyon", index=True
            )

        # 6. Temizleme_Logu
        if result.cleaning_log:
            log_rows = [
                {
                    "Sutun": a.column,
                    "Sorun": a.issue,
                    "Seviye": a.level.value,
                    "Uygulandi": a.applied,
                    "Detay": a.detail,
                }
                for a in result.cleaning_log
            ]
            df_cleaning = pd.DataFrame(log_rows)
        else:
            df_cleaning = pd.DataFrame(columns=["Sutun", "Sorun", "Seviye", "Uygulandi", "Detay"])
        df_cleaning.to_excel(writer, sheet_name="Temizleme_Logu", index=False)

        # 7. PII_Uyarilari
        if result.pii_findings:
            pii_rows = [
                {
                    "Sutun": f.column,
                    "PII Turu": f.pii_type.value,
                    "Guven": f.confidence,
                    "Eslesme Orani": f.match_rate,
                    "Ornek Sayisi": f.sample_count,
                    "Tespit Katmani": f.detection_layer,
                }
                for f in result.pii_findings
            ]
            df_pii = pd.DataFrame(pii_rows)
        else:
            df_pii = pd.DataFrame(
                columns=["Sutun", "PII Turu", "Guven", "Eslesme Orani", "Ornek Sayisi", "Tespit Katmani"]
            )
        df_pii.to_excel(writer, sheet_name="PII_Uyarilari", index=False)

    # ── Apply formatting ──────────────────────────────────────────────────────
    _format_workbook(out_path)

    return out_path


def write_excel(result: ReportResult, config: PipelineConfig, out_path: str) -> str:
    """Thin alias for render_excel (mirrors html_reporter's write_html pattern)."""
    return render_excel(result, config, out_path)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _format_workbook(path: str) -> None:
    """Apply basic formatting: bold header, freeze first row, auto-fit column widths."""
    from openpyxl import load_workbook
    from openpyxl.styles import Font

    wb = load_workbook(path)
    for ws in wb.worksheets:
        # Bold header row
        bold_font = Font(bold=True)
        for cell in ws[1]:
            if cell.value is not None:
                cell.font = bold_font

        # Freeze first row
        ws.freeze_panes = "A2"

        # Auto-fit column widths (capped between 10 and 60)
        from openpyxl.utils import get_column_letter

        for col_idx, col_cells in enumerate(ws.columns, start=1):
            max_len = 0
            for cell in col_cells:
                if cell.value is not None:
                    val_str = str(cell.value)
                    # Approximate: wider chars count as 2
                    length = sum(2 if ord(c) > 127 else 1 for c in val_str)
                    if length > max_len:
                        max_len = length
            col_letter = get_column_letter(col_idx)
            width = min(max(max_len + 3, 10), 60)
            ws.column_dimensions[col_letter].width = width

    wb.save(path)
