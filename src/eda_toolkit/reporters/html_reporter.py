from __future__ import annotations

import plotly.offline
import pandas as pd

from eda_toolkit.types import ReportResult, PipelineConfig
from eda_toolkit.viz import (
    fig_correlation_heatmap,
    fig_histogram,
    fig_cluster_scatter,
    fig_cluster_bar,
)


def render_html(result: ReportResult, config: PipelineConfig) -> str:
    """Build a standalone single-file HTML report with Plotly JS embedded.

    The report includes:
    - Turkish title "Veri Analiz Raporu"
    - Summary table (row/column counts)
    - Missing-summary table
    - PII alerts (when findings present)
    - One Plotly histogram per numeric column
    - Correlation heatmap (when correlation_matrix present)
    - Outliers table (when outliers dict present)
    - Hypothesis tests table (when hypothesis_tests list present)
    - Clustering visualization (when cluster_labels present)

    Returns the full HTML string.
    """
    df = result.cleaned_df
    if df is None:
        raise ValueError("ReportResult.cleaned_df is None; cannot render HTML report.")

    # Collect HTML parts
    parts: list[str] = []

    # Header
    parts.append('<!DOCTYPE html>\n<html lang="tr">\n<head>')
    parts.append('<meta charset="utf-8">')
    parts.append('<title>Veri Analiz Raporu</title>')
    parts.append("""
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 960px; margin: 0 auto; padding: 20px; background: #f9fafb; }
h1 { color: #1e3a5f; border-bottom: 3px solid #1e3a5f; padding-bottom: 8px; }
h2 { color: #2c5282; margin-top: 32px; }
table { border-collapse: collapse; width: 100%%; margin: 12px 0; background: white; }
th, td { border: 1px solid #d1d5db; padding: 8px 12px; text-align: left; }
th { background: #e5e7eb; font-weight: 600; }
tr:nth-child(even) { background: #f3f4f6; }
.chart { margin: 24px 0; background: white; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.warning { background: #fef3c7; border: 1px solid #f59e0b; padding: 12px; border-radius: 6px; margin: 12px 0; }
.warning p { margin: 4px 0; color: #92400e; }
.info { color: #6b7280; font-style: italic; }
</style>
""")
    # Embed Plotly JS inline in <head> (offline-compatible, loads before all chart scripts)
    parts.append(_get_plotly_js_inline())
    parts.append("</head>\n<body>")

    # Title
    parts.append("<h1>Veri Analiz Raporu</h1>")

    # Summary table
    parts.append("<h2>Ozet Bilgiler</h2>")
    parts.append("<table>")
    parts.append(f"<tr><th>Dosya</th><td>{result.input_path}</td></tr>")
    parts.append(f"<tr><th>Satir Sayisi</th><td>{result.row_count}</td></tr>")
    parts.append(f"<tr><th>Sutun Sayisi</th><td>{result.col_count}</td></tr>")
    parts.append("</table>")

    # Missing summary
    parts.append("<h2>Eksik Veri Ozeti</h2>")
    missing_df: pd.DataFrame = result.eda.missing_summary
    if not missing_df.empty:
        parts.append(missing_df.to_html(index=False, border=0))
    else:
        parts.append("<p>Eksik veri bulunamadi.</p>")

    # ── PII Uyarilari ──────────────────────────────────────────────────────────
    parts.append("<h2>PII Uyarilari</h2>")
    if result.pii_findings:
        parts.append('<div class="warning">')
        parts.append("<p>Bu veri kumesinde kisisel veri icerebilecek sutunlar tespit edildi.</p>")
        parts.append("</div>")
        parts.append("<table>")
        parts.append("<tr><th>Sutun</th><th>Tur</th><th>Guven (%)</th><th>Eslesme Orani (%)</th><th>Katman</th></tr>")
        for f in result.pii_findings:
            parts.append(
                f"<tr><td>{f.column}</td><td>{f.pii_type.value}</td>"
                f"<td>{f.confidence * 100:.1f}</td><td>{f.match_rate * 100:.1f}</td>"
                f"<td>{f.detection_layer}</td></tr>"
            )
        parts.append("</table>")
    else:
        parts.append('<p class="info">Kisisel veri (PII) tespit edilmedi.</p>')

    # ── Correlation Heatmap ────────────────────────────────────────────────────
    corr = result.eda.correlation_matrix
    if corr is not None and not corr.empty and corr.shape[0] >= 2:
        parts.append("<h2>Korelasyon Matrisi</h2>")
        fig_corr = fig_correlation_heatmap(corr)
        parts.append(f'<div class="chart">{fig_corr.to_html(full_html=False, include_plotlyjs=False)}</div>')

    # ── Outliers Table ─────────────────────────────────────────────────────────
    outliers = result.eda.outliers
    if outliers is not None and len(outliers) > 0:
        parts.append("<h2>Aykiri Degerler</h2>")
        parts.append("<table>")
        parts.append("<tr><th>Sutun</th><th>Aykiri Adedi</th><th>Oran</th><th>Alt Sinir</th><th>Ust Sinir</th></tr>")
        for col_name, info in outliers.items():
            cnt = info.get("count", 0)
            rate = info.get("rate", 0.0)
            lower = info.get("lower")
            upper = info.get("upper")
            lower_str = f"{lower:.2f}" if lower is not None else "-"
            upper_str = f"{upper:.2f}" if upper is not None else "-"
            parts.append(
                f"<tr><td>{col_name}</td><td>{cnt}</td><td>{rate:.4f}</td>"
                f"<td>{lower_str}</td><td>{upper_str}</td></tr>"
            )
        parts.append("</table>")

    # ── Hypothesis Tests ───────────────────────────────────────────────────────
    hyp_tests = result.eda.hypothesis_tests
    if hyp_tests is not None and len(hyp_tests) > 0:
        parts.append("<h2>Hipotez Testleri</h2>")
        parts.append("<table>")
        parts.append("<tr><th>Test Turu</th><th>Sutun(lar)</th><th>Istatistik</th><th>p-degeri</th><th>Sonuc</th></tr>")
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

            parts.append(
                f"<tr><td>{test_label}</td><td>{col_str}</td>"
                f"<td>{stat_str}</td><td>{p_str}</td><td>{sonuc}</td></tr>"
            )
        parts.append("</table>")

    # ── Charts: one histogram per numeric column ───────────────────────────────
    parts.append("<h2>Sayisal Sutun Dagilimlari</h2>")
    numeric_cols = df.select_dtypes(include=["number"])
    for col in numeric_cols.columns:
        series = df[col].dropna()
        fig = fig_histogram(series, col)
        chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
        parts.append(f'<div class="chart">{chart_html}</div>')

    # ── Clustering Visualization ───────────────────────────────────────────────
    cluster_labels = result.eda.cluster_labels
    if cluster_labels is not None and len(cluster_labels) > 0:
        parts.append("<h2>Kumeleme Sonuclari</h2>")
        numeric_cols_clust = df.select_dtypes(include=["number"])
        if numeric_cols_clust.shape[1] >= 2:
            # Scatter plot: first two numeric columns colored by cluster label
            plot_df = df.copy()
            plot_df["_cluster"] = cluster_labels.values
            # Filter out unclustered points (label == -1)
            valid_mask = plot_df["_cluster"] != -1
            plot_df_valid = plot_df[valid_mask]
            col_x = numeric_cols_clust.columns[0]
            col_y = numeric_cols_clust.columns[1]
            fig_clust = fig_cluster_scatter(plot_df_valid, col_x, col_y, plot_df)
            parts.append(f'<div class="chart">{fig_clust.to_html(full_html=False, include_plotlyjs=False)}</div>')
        else:
            # Single or no numeric column: bar chart of cluster counts
            fig_bar = fig_cluster_bar(cluster_labels)
            parts.append(f'<div class="chart">{fig_bar.to_html(full_html=False, include_plotlyjs=False)}</div>')

    # Close body + html
    parts.append("</body>\n</html>")

    return "\n".join(parts)


def write_html(result: ReportResult, config: PipelineConfig, out_path: str) -> None:
    """Render HTML report and write to out_path."""
    html = render_html(result, config)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def _get_plotly_js_inline() -> str:
    """Return a <script> tag with Plotly JS inlined for offline use."""
    return f"<script>{plotly.offline.get_plotlyjs()}</script>"
