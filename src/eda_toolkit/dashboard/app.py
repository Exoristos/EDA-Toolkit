"""Streamlit dashboard for the eda_toolkit pipeline.

Single-file app. Sidebar holds the pipeline configuration; the main area is
organised into eight tabs (Ozet, Tanimlayici Istatistik, Gorseller, Ileri
Analiz, Tahmin, Modelleme, PII Uyarilari, Raporlar). User-facing text is
Turkish; identifiers are English. No emojis.

Pure helper functions live at the top of the module so they can be unit-tested
without a Streamlit runtime.
"""

from __future__ import annotations

import math
import tempfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eda_toolkit import pipeline
from eda_toolkit.analytics.modeling import fit_regression
from eda_toolkit.analytics.timeseries import forecast
from eda_toolkit.pii.masker import mask_dataframe
from eda_toolkit.types import (
    CleaningAction,
    CleaningLevel,
    ColumnSchema,
    MaskMode,
    PiiFinding,
    PiiMode,
    PipelineConfig,
    ReportResult,
)
from eda_toolkit.viz import (
    fig_cluster_bar,
    fig_cluster_scatter,
    fig_correlation_heatmap,
    fig_histogram,
    fig_missing_bar,
)

# ── Theme constants (deep navy primary, amber warnings) ───────────────────────

_THEME_CSS = """
<style>
    /* App shell */
    .stApp { background: #f7f8fa; }
    /* Header band */
    .eda-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        padding: 1.1rem 1.6rem 1.15rem;
        border-radius: 0 0 12px 12px;
        margin: -1.2rem -1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(30, 58, 95, 0.18);
    }
    .eda-header h1 {
        color: #ffffff; font-size: 1.5rem; font-weight: 700;
        margin: 0; letter-spacing: 0.2px;
    }
    .eda-header p {
        color: #c7d4e6; font-size: 0.88rem; margin: 0.25rem 0 0; font-weight: 400;
    }
    /* Section labels inside tabs */
    .eda-section {
        font-size: 1.02rem; font-weight: 600; color: #1e3a5f;
        border-left: 3px solid #1e3a5f; padding-left: 0.6rem;
        margin: 0.4rem 0 0.9rem;
    }
    .eda-quiet { color: #6b7280; font-size: 0.85rem; }
    /* Metric value emphasis */
    [data-testid="stMetricValue"] { color: #1e3a5f; font-weight: 700; }
    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #1e3a5f;
    }
    .eda-sidebar-title {
        font-size: 1.05rem; font-weight: 700; color: #1e3a5f;
        margin: 0 0 0.2rem;
    }
    .eda-sidebar-sub { color: #6b7280; font-size: 0.8rem; margin: 0 0 0.8rem; }
    .eda-divider { border-top: 1px solid #e5e7eb; margin: 1rem 0; }
</style>
"""

# ── TR label <-> enum maps ────────────────────────────────────────────────────

_CLEANING_LABELS: list[str] = ["Konservatif", "Orta", "Saldırgan"]
_CLEANING_HELP: dict[str, str] = {
    "Konservatif": "Veriye dokunma; yalnizca tespit.",
    "Orta": "Guvenli, geri donulebilir duzeltmeler.",
    "Saldırgan": "Imputasyon + aykiri deger isleme uygulanir.",
}
_CLEANING_MAP: dict[str, CleaningLevel] = {
    "Konservatif": CleaningLevel.CONSERVATIVE,
    "Orta": CleaningLevel.MODERATE,
    "Saldırgan": CleaningLevel.AGGRESSIVE,
}

_PII_LABELS: list[str] = ["Uyar", "Maske"]
_PII_MAP: dict[str, PiiMode] = {"Uyar": PiiMode.WARN, "Maske": PiiMode.MASK}

_MASK_LABELS: list[str] = ["Kısmi", "Tam", "Hash", "Tokenize", "Pseudonimleştir"]
_MASK_MAP: dict[str, MaskMode] = {
    "Kısmi": MaskMode.PARTIAL,
    "Tam": MaskMode.FULL,
    "Hash": MaskMode.HASH,
    "Tokenize": MaskMode.TOKENIZE,
    "Pseudonimleştir": MaskMode.PSEUDONYMIZE,
}
_MASK_LABEL_OF: dict[MaskMode, str] = {v: k for k, v in _MASK_MAP.items()}


# ── Pure helper functions (no Streamlit dependency) ───────────────────────────
# These convert pipeline outputs into display-friendly DataFrames and apply
# per-column masking. Importable and unit-testable without a Streamlit runtime.


def schemas_to_dataframe(schemas: list[ColumnSchema]) -> pd.DataFrame:
    """Build a display DataFrame from a list of ColumnSchema objects.

    Columns: name / dtype / kind / unique_count / missing_count.
    Returns an empty (header-only) DataFrame when the list is empty.
    """
    rows = [
        {
            "Sutun": s.name,
            "Tip": s.dtype,
            "Sinif": s.kind,
            "Essiz Deger": s.unique_count,
            "Eksik": s.missing_count,
        }
        for s in (schemas or [])
    ]
    if not rows:
        return pd.DataFrame(columns=["Sutun", "Tip", "Sinif", "Essiz Deger", "Eksik"])
    return pd.DataFrame(rows)


def cleaning_log_to_dataframe(log: list[CleaningAction]) -> pd.DataFrame:
    """Build a display DataFrame from a list of CleaningAction objects.

    Columns: column / issue / level / applied / detail.
    """
    rows = [
        {
            "Sutun": a.column,
            "Sorun": a.issue,
            "Seviye": a.level.value if isinstance(a.level, CleaningLevel) else str(a.level),
            "Uygulandi": "Evet" if a.applied else "Hayir",
            "Detay": a.detail,
        }
        for a in (log or [])
    ]
    if not rows:
        return pd.DataFrame(columns=["Sutun", "Sorun", "Seviye", "Uygulandi", "Detay"])
    return pd.DataFrame(rows)


def outliers_to_dataframe(outliers: dict | None) -> pd.DataFrame:
    """Convert the advanced-EDA outliers dict into a display DataFrame.

    Columns: Sutun / Aykiri Adet / Oran / Alt Sinir / Ust Sinir.
    Missing bounds are rendered as "-".
    """
    if not outliers:
        return pd.DataFrame(columns=["Sutun", "Aykiri Adet", "Oran", "Alt Sinir", "Ust Sinir"])
    rows = []
    for col, info in outliers.items():
        info = info or {}
        lower = info.get("lower")
        upper = info.get("upper")
        rows.append({
            "Sutun": col,
            "Aykiri Adet": info.get("count", 0),
            "Oran": info.get("rate", 0.0),
            "Alt Sinir": f"{lower:.2f}" if lower is not None else "-",
            "Ust Sinir": f"{upper:.2f}" if upper is not None else "-",
        })
    return pd.DataFrame(rows)


def hypothesis_tests_to_dataframe(tests: list | None) -> pd.DataFrame:
    """Convert the advanced-EDA hypothesis-tests list into a display DataFrame.

    Columns: Test / Sutun(lar) / Istatistik / p-degeri / Sonuc.
    Handles both ``shapiro`` (column / is_normal) and ``pearson`` (columns /
    significant) result shapes, plus an optional ``note`` field.
    """
    cols = ["Test", "Sutun(lar)", "Istatistik", "p-degeri", "Sonuc"]
    if not tests:
        return pd.DataFrame(columns=cols)
    rows = []
    for t in tests:
        test_type = t.get("test", "")
        if test_type == "shapiro":
            label = "Shapiro-Wilk"
            col_str = str(t.get("column", ""))
            is_normal = t.get("is_normal")
            sonuc = (
                "Normal" if is_normal is True
                else "Normal degil" if is_normal is False
                else "Belirsiz"
            )
        elif test_type == "pearson":
            label = "Pearson"
            cols_list = t.get("columns", [])
            col_str = ", ".join(cols_list) if cols_list else ""
            is_sig = t.get("significant")
            sonuc = (
                "Anlamli" if is_sig is True
                else "Anlamsiz" if is_sig is False
                else "Belirsiz"
            )
        else:
            label = test_type or "-"
            col_str = str(t.get("column", t.get("columns", "")))
            sonuc = t.get("note", "")

        # note supplement (e.g. "Sabit sutun" or "Test basarisiz") when no verdict
        if sonuc == "" or sonuc == "Belirsiz":
            note = t.get("note")
            if note:
                sonuc = f"{sonuc} ({note})" if sonuc else note

        stat = t.get("statistic")
        p_val = t.get("p_value")
        rows.append({
            "Test": label,
            "Sutun(lar)": col_str,
            "Istatistik": f"{stat:.4f}" if stat is not None else "-",
            "p-degeri": f"{p_val:.4f}" if p_val is not None else "-",
            "Sonuc": sonuc or "-",
        })
    return pd.DataFrame(rows, columns=cols)


def findings_to_dataframe(findings: list[PiiFinding]) -> pd.DataFrame:
    """Convert a list of PiiFinding objects into a display DataFrame.

    Columns: Sutun / Tur / Guven (%) / Eslesme (%) / Katman.
    """
    cols = ["Sutun", "Tur", "Guven (%)", "Eslesme (%)", "Katman"]
    if not findings:
        return pd.DataFrame(columns=cols)
    rows = [
        {
            "Sutun": f.column,
            "Tur": f.pii_type.value,
            "Guven (%)": round(f.confidence * 100, 1),
            "Eslesme (%)": round(f.match_rate * 100, 1),
            "Katman": f.detection_layer,
        }
        for f in findings
    ]
    return pd.DataFrame(rows, columns=cols)


def apply_per_column_mask(
    df: pd.DataFrame,
    findings: list[PiiFinding],
    column_modes: dict[str, MaskMode],
    show_last: int = 4,
) -> pd.DataFrame:
    """Apply masking column-by-column using a per-column mask mode.

    ``mask_dataframe`` accepts a single global mode, so we mask each PII column
    by calling it on a single-column slice filtered to that column's finding.
    The input DataFrame is never mutated; a fresh copy is returned. Columns
    missing from ``column_modes`` fall back to ``MaskMode.PARTIAL``.
    """
    masked = df.copy()
    for finding in findings:
        col = finding.column
        if col not in masked.columns:
            continue
        mode = column_modes.get(col, MaskMode.PARTIAL)
        single_masked = mask_dataframe(
            masked[[col]], [finding], mode, show_last=show_last,
        )
        masked[col] = single_masked[col]
    return masked


def _fmt_num(value, precision: int = 4) -> str:
    """Format a numeric value for display; non-numerics/None become ``-``."""
    if value is None:
        return "-"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f) or math.isinf(f):
        return "-"
    return f"{f:.{precision}f}"


def quality_score_to_dataframe(qs: dict | None) -> pd.DataFrame:
    """Build a compact breakdown table for ``data_quality_score`` output.

    Columns: Bilesen / Deger / Skora Etkisi. Returns an empty frame on None.
    Pure (no Streamlit) so it can be unit-tested directly.
    """
    cols = ["Bilesen", "Deger", "Skora Etkisi"]
    if not qs:
        return pd.DataFrame(columns=cols)
    comp = qs.get("components", {}) or {}
    rows = [
        {"Bilesen": "Toplam Kalite Skoru", "Deger": f"{qs.get('score', 0):.1f}/100", "Skora Etkisi": "-"},
        {"Bilesen": "Eksik Veri Orani", "Deger": f"{qs.get('missing_rate', 0):.1%}", "Skora Etkisi": f"-{comp.get('missing_penalty', 0):.1f}"},
        {"Bilesen": "Tekrarli Satir Orani", "Deger": f"{qs.get('duplicate_rate', 0):.1%}", "Skora Etkisi": f"-{comp.get('duplicate_penalty', 0):.1f}"},
        {"Bilesen": "Aykiri Deger Orani", "Deger": f"{qs.get('outlier_rate', 0):.1%}", "Skora Etkisi": f"-{comp.get('outlier_penalty', 0):.1f}"},
        {"Bilesen": "Sabit Sutun Sayisi", "Deger": int(qs.get("constant_columns", 0)), "Skora Etkisi": f"-{comp.get('constant_penalty', 0):.1f}"},
    ]
    return pd.DataFrame(rows, columns=cols)


def risk_score_to_dataframe(rs: dict | None) -> pd.DataFrame:
    """Build a compact breakdown table for ``pii_risk_score`` output.

    Columns: Bilesen / Katki. Returns an empty frame on None.
    Pure (no Streamlit).
    """
    cols = ["Bilesen", "Katki"]
    if not rs:
        return pd.DataFrame(columns=cols)
    factors = rs.get("factors", {}) or {}
    sev = factors.get("severity", {}) or {}
    card = factors.get("cardinality", {}) or {}
    qi = factors.get("qi_combinability", {}) or {}
    rows = [
        {"Bilesen": "Toplam Risk Skoru", "Katki": f"{rs.get('score', 0):.1f}/100"},
        {"Bilesen": f"PII Ciddiyeti ({sev.get('max_type') or '-'})", "Katki": f"{sev.get('score', 0):.1f}"},
        {"Bilesen": "Kardinalite", "Katki": f"{card.get('score', 0):.1f}"},
        {"Bilesen": "QI Birlesim", "Katki": f"{qi.get('score', 0):.1f}"},
        {"Bilesen": "Maskeleme Katkisi", "Katki": f"{factors.get('masking_contribution', 0):.1f}"},
    ]
    k = rs.get("k_anonymity")
    if k is not None:
        rows.append({"Bilesen": "k-anonimlik", "Katki": str(k)})
    return pd.DataFrame(rows, columns=cols)


def regression_result_to_dataframe(res: dict | None) -> pd.DataFrame:
    """Build a coefficients table (Degisken / Katsayi / p-degeri / VIF).

    Pure (no Streamlit). Returns an empty frame when the result is missing or
    carries an ``error`` key instead of a coefficients map.
    """
    cols = ["Degisken", "Katsayi", "p-degeri", "VIF"]
    if not res or "error" in res or "coefficients" not in res:
        return pd.DataFrame(columns=cols)
    coefs = res.get("coefficients", {}) or {}
    pvals = res.get("p_values", {}) or {}
    vifs = res.get("vif", {}) or {}
    rows = []
    for name, coef in coefs.items():
        rows.append({
            "Degisken": name,
            "Katsayi": _fmt_num(coef, 4),
            "p-degeri": _fmt_num(pvals.get(name), 4),
            "VIF": _fmt_num(vifs.get(name), 2),
        })
    return pd.DataFrame(rows, columns=cols)


def fig_forecast(fc: dict, col: str, history: pd.Series | None = None) -> go.Figure | None:
    """Build a Plotly line chart for a ``forecast()`` result.

    Plots observed history (if given) as a solid navy line and the forecast
    points as a dashed line with a shaded %95 confidence band. Returns ``None``
    when the result carries an error or has no points. Matches ``viz.py`` style
    (plotly_white template, navy accents). Pure — no Streamlit.
    """
    if not fc or "error" in fc:
        return None
    points = fc.get("point") or []
    ci_lower = fc.get("ci_lower") or []
    ci_upper = fc.get("ci_upper") or []
    if not points:
        return None

    hist_y = list(history.values) if history is not None else []
    n_hist = len(hist_y)
    fc_x = list(range(n_hist, n_hist + len(points)))
    hist_x = list(range(n_hist))

    fig = go.Figure()
    if hist_y:
        fig.add_trace(go.Scatter(
            x=hist_x, y=hist_y, mode="lines", name="Gozlemlenen",
            line=dict(color="#1e3a5f"),
        ))
    fig.add_trace(go.Scatter(
        x=fc_x, y=points, mode="lines+markers", name="Tahmin",
        line=dict(color="#2c5282", dash="dash"),
    ))
    if len(ci_lower) == len(points) and len(ci_upper) == len(points):
        fig.add_trace(go.Scatter(
            x=fc_x + fc_x[::-1],
            y=ci_upper + ci_lower[::-1],
            fill="toself", fillcolor="rgba(44,82,130,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="%95 Guven Araligi", hoverinfo="skip",
        ))

    fig.update_layout(
        title=f"{col} - Tahmin",
        template="plotly_white",
        xaxis_title="Adim",
        yaxis_title=col,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


# ── Streamlit render layer ────────────────────────────────────────────────────


def _render_header() -> None:
    st.markdown(
        """
        <div class="eda-header">
            <h1>Veri Analiz Paneli</h1>
            <p>Dosya yukleyin, temizleme ve PII tercihlerini yapin, analiz edin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section(label: str) -> None:
    st.markdown(f'<div class="eda-section">{label}</div>', unsafe_allow_html=True)


def _render_sidebar() -> dict:
    """Render the sidebar configuration controls. Returns the chosen config."""
    with st.sidebar:
        st.markdown('<div class="eda-sidebar-title">Yapilandirma</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="eda-sidebar-sub">Analiz secenekleri ve PII ayarlari</div>',
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader("Veri dosyasi", type=["csv", "xlsx", "xls"])

        cleaning_label = st.radio(
            "Temizleme Seviyesi",
            _CLEANING_LABELS,
            help=" | ".join(f"{k}: {v}" for k, v in _CLEANING_HELP.items()),
        )
        cleaning_level = _CLEANING_MAP[cleaning_label]
        with st.expander("Seviye aciklamalari", expanded=False):
            for k in _CLEANING_LABELS:
                st.markdown(f"**{k}** &mdash; {_CLEANING_HELP[k]}", unsafe_allow_html=True)

        pii_label = st.radio("PII Modu", _PII_LABELS)
        pii_mode = _PII_MAP[pii_label]

        mask_mode = MaskMode.PARTIAL
        show_last = 4
        if pii_mode == PiiMode.MASK:
            mask_label = st.selectbox("Maskeleme Modu", _MASK_LABELS)
            mask_mode = _MASK_MAP[mask_label]
            show_last = st.slider("Gosterilecek son hane", 0, 8, 4)

        st.markdown('<div class="eda-divider"></div>', unsafe_allow_html=True)
        run_clicked = st.button("Analiz Et", type="primary", use_container_width=True)

    return {
        "uploaded": uploaded,
        "cleaning_level": cleaning_level,
        "pii_mode": pii_mode,
        "mask_mode": mask_mode,
        "show_last": show_last,
        "run_clicked": run_clicked,
    }


# ── Tab renderers ─────────────────────────────────────────────────────────────


def _tab_summary(result: ReportResult, original_name: str) -> None:
    _section("Dosya Ozeti")
    c1, c2, c3 = st.columns(3)
    c1.metric("Dosya", original_name)
    c2.metric("Satir Sayisi", result.row_count)
    c3.metric("Sutun Sayisi", result.col_count)

    _section("Veri Kalitesi ve Gizlilik Riski")
    qs = result.eda.quality_score
    rs = result.eda.risk_score
    if qs is None and rs is None:
        st.markdown(
            '<div class="eda-quiet">Skorlar hesaplanamadi.</div>',
            unsafe_allow_html=True,
        )
    else:
        left, right = st.columns([1, 1])
        with left:
            if qs is not None:
                st.metric("Kalite Skoru", f"{qs.get('score', 0):.0f}")
                st.markdown(
                    '<div class="eda-quiet">Eksik, tekrarli ve aykiri degerlere dayali genel kalite puani (0-100).</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    quality_score_to_dataframe(qs),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.markdown(
                    '<div class="eda-quiet">Kalite skoru hesaplanamadi.</div>',
                    unsafe_allow_html=True,
                )
        with right:
            if rs is not None:
                risk_score = float(rs.get("score", 0))
                st.metric("Risk Skoru", f"{risk_score:.0f}")
                if risk_score >= 70:
                    st.warning(
                        "Yeniden tanimlanma riski yuksek. PII maskeleme duzeyini "
                        "artirmayi veya quasi-tanimlayici sutunlari daraltmayi degerlendirin."
                    )
                else:
                    st.markdown(
                        '<div class="eda-quiet">PII bulgularina dayali yeniden tanimlanma riski puani (0-100).</div>',
                        unsafe_allow_html=True,
                    )
                st.dataframe(
                    risk_score_to_dataframe(rs),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.markdown(
                    '<div class="eda-quiet">Risk skoru hesaplanamadi.</div>',
                    unsafe_allow_html=True,
                )

    _section("Sutun Semasi")
    schema_df = schemas_to_dataframe(result.eda.schemas)
    st.dataframe(schema_df, use_container_width=True, hide_index=True)

    _section("Temizleme Islemleri")
    cleaning_df = cleaning_log_to_dataframe(result.cleaning_log)
    if cleaning_df.empty:
        st.markdown(
            '<div class="eda-quiet">Temizleme islemi uygulanmadi.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.dataframe(cleaning_df, use_container_width=True, hide_index=True)


def _tab_descriptive(result: ReportResult) -> None:
    _section("Tanimlayici Istatistikler")
    if result.eda.descriptive is None or result.eda.descriptive.empty:
        st.markdown(
            '<div class="eda-quiet">Sayisal sutun bulunamadi.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.dataframe(result.eda.descriptive, use_container_width=True)

    _section("Eksik Veri")
    missing = result.eda.missing_summary
    if missing is None or missing.empty:
        st.markdown(
            '<div class="eda-quiet">Eksik veri bulunamadi.</div>',
            unsafe_allow_html=True,
        )
    else:
        left, right = st.columns([1, 1])
        with left:
            st.dataframe(missing, use_container_width=True, hide_index=True)
        with right:
            fig = fig_missing_bar(missing)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown(
                    '<div class="eda-quiet">Eksik oranina sahip sutun yok.</div>',
                    unsafe_allow_html=True,
                )


def _tab_charts(result: ReportResult) -> None:
    _section("Korelasyon Matrisi")
    corr = result.eda.correlation_matrix
    if corr is not None and not corr.empty and corr.shape[0] >= 2:
        st.plotly_chart(fig_correlation_heatmap(corr), use_container_width=True)
    else:
        st.markdown(
            '<div class="eda-quiet">Korelasyon hesaplanamadi (2+ sayisal sutun gerekir).</div>',
            unsafe_allow_html=True,
        )

    _section("Sayisal Sutun Dagilimlari")
    df = result.cleaned_df
    if df is None or df.select_dtypes(include=["number"]).shape[1] == 0:
        st.markdown(
            '<div class="eda-quiet">Histogram cizilecek sayisal sutun yok.</div>',
            unsafe_allow_html=True,
        )
        return
    numeric_cols = df.select_dtypes(include=["number"])
    for col in numeric_cols.columns:
        series = df[col].dropna()
        st.plotly_chart(fig_histogram(series, col), use_container_width=True)


def _tab_advanced(result: ReportResult) -> None:
    _section("Aykiri Degerler")
    outliers_df = outliers_to_dataframe(result.eda.outliers)
    if outliers_df.empty:
        st.markdown(
            '<div class="eda-quiet">Aykiri deger analizi uygulanmadi.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.dataframe(outliers_df, use_container_width=True, hide_index=True)

    _section("Hipotez Testleri")
    hyp_df = hypothesis_tests_to_dataframe(result.eda.hypothesis_tests)
    if hyp_df.empty:
        st.markdown(
            '<div class="eda-quiet">Hipotez testi sonucu bulunamadi.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.dataframe(hyp_df, use_container_width=True, hide_index=True)

    _section("Kumeleme")
    cluster_labels = result.eda.cluster_labels
    df = result.cleaned_df
    if cluster_labels is None or len(cluster_labels) == 0:
        st.markdown(
            '<div class="eda-quiet">Kumeleme sonucu bulunamadi.</div>',
            unsafe_allow_html=True,
        )
        return

    numeric_cols = df.select_dtypes(include=["number"]) if df is not None else pd.DataFrame()
    if numeric_cols.shape[1] >= 2:
        plot_df = df.copy()
        plot_df["_cluster"] = cluster_labels.values
        valid_mask = plot_df["_cluster"] != -1
        plot_df_valid = plot_df[valid_mask]
        col_x = numeric_cols.columns[0]
        col_y = numeric_cols.columns[1]
        unclustered_n = int((~valid_mask).sum())
        if unclustered_n > 0:
            st.markdown(
                f'<div class="eda-quiet">{unclustered_n} satir kumelendirilemedi (-1).</div>',
                unsafe_allow_html=True,
            )
        fig = fig_cluster_scatter(plot_df_valid, col_x, col_y, fallback_plot_df=plot_df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(fig_cluster_bar(cluster_labels), use_container_width=True)


def _safe_source_df(result: ReportResult) -> pd.DataFrame | None:
    """Return the unmasked cleaned df for interactive modeling/forecast.

    Falls back to ``cleaned_df`` (which may be masked under PII==MASK).
    """
    if result.unmasked_cleaned_df is not None:
        return result.unmasked_cleaned_df
    return result.cleaned_df


def _tab_forecast(result: ReportResult) -> None:
    _section("Zaman Serisi Tahmini")
    df = _safe_source_df(result)
    if df is None or df.empty:
        st.markdown(
            '<div class="eda-quiet">Tahmin icin veri bulunamadi.</div>',
            unsafe_allow_html=True,
        )
        return

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        st.markdown(
            '<div class="eda-quiet">Sayisal sutun bulunamadi; tahmin yapilamaz.</div>',
            unsafe_allow_html=True,
        )
        return

    col = st.selectbox("Hedef sayisal sutun", numeric_cols, key="fc_col")
    horizon = st.slider("Tahmin ufku (adim)", 3, 24, 6, key="fc_horizon")
    method_label = st.radio("Yontem", ["ARIMA", "SARIMAX"], key="fc_method")
    period: int | None = None
    if method_label == "SARIMAX":
        period = st.number_input(
            "Mevsim periyodu", min_value=2, max_value=52,
            value=12, step=1, key="fc_period",
        )

    if st.button("Tahmin Yap", type="primary", key="fc_btn"):
        method = method_label.lower()
        try:
            with st.spinner("Tahmin hesaplaniyor..."):
                fc = forecast(
                    df, col, int(horizon),
                    method=method,
                    period=period if method == "sarimax" else None,
                )
        except Exception as exc:  # noqa: BLE001 - defensive; never crash the tab
            st.error(f"Tahmin hatasi: {exc}")
            return
        _render_forecast_result(fc, col, df[col])


def _render_forecast_result(fc: dict, col: str, history: pd.Series) -> None:
    """Render the forecast dict: error / warning / chart + metadata caption."""
    if not fc or "error" in fc:
        st.error(fc.get("error", "Tahmin uretilemedi."))
        return

    warning = fc.get("warning")
    if warning:
        st.warning(warning)

    fig = fig_forecast(fc, col, history)
    if fig is None:
        st.markdown(
            '<div class="eda-quiet">Tahmin noktasi uretilemedi.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="eda-quiet">Yontem: '
        f'{(fc.get("method") or "-").upper()} | '
        f'Duzen: {fc.get("order", "-")} | '
        f'Ufuk: {fc.get("horizon", "-")} adim</div>',
        unsafe_allow_html=True,
    )


def _tab_modeling(result: ReportResult) -> None:
    _section("Regresyon Modelleme")
    df = _safe_source_df(result)
    if df is None or df.empty:
        st.markdown(
            '<div class="eda-quiet">Modelleme icin veri bulunamadi.</div>',
            unsafe_allow_html=True,
        )
        return

    cols = df.columns.tolist()
    target = st.selectbox("Hedef sutun", cols, key="md_target")
    feature_options = [c for c in cols if c != target]
    features = st.multiselect(
        "Bagimsiz degiskenler", feature_options, key="md_features",
    )
    kind_label = st.radio("Model turu", ["Lineer", "Lojistik"], key="md_kind")

    if st.button("Modeli Egit", type="primary", key="md_btn"):
        if not features:
            st.warning("En az bir bagimsiz degisken secin.")
            return
        kind = "linear" if kind_label == "Lineer" else "logistic"
        try:
            with st.spinner("Model egitiliyor..."):
                res = fit_regression(df, target, features, kind=kind)
        except Exception as exc:  # noqa: BLE001 - defensive
            st.error(f"Model egitim hatasi: {exc}")
            return
        _render_modeling_result(res)


def _render_modeling_result(res: dict) -> None:
    """Render the regression result dict: error / metrics / VIF / coefficients."""
    if not res or "error" in res:
        st.error(res.get("error", "Model egitilemedi."))
        return

    c1, c2, c3 = st.columns(3)
    if "r2" in res:
        c1.metric("R-kare", _fmt_num(res["r2"], 3))
    elif "pseudo_r2" in res:
        c1.metric("Pseudo R-kare", _fmt_num(res["pseudo_r2"], 3))
    else:
        c1.metric("R-kare", "-")
    if "accuracy" in res:
        c2.metric("Dogruluk", f"{float(res['accuracy']):.1%}")
    else:
        c2.metric("Dogruluk", "-")
    c3.metric("Gozlem", res.get("n", "-"))

    vif_warn = res.get("vif_warning") or []
    if vif_warn:
        st.warning(
            "Yuksek VIF (>10) tespit edilen degiskenler: "
            + ", ".join(vif_warn)
            + ". Bu degiskenler arasinda coklu dogrusal baglanti olabilir."
        )

    coef_df = regression_result_to_dataframe(res)
    if coef_df.empty:
        st.markdown(
            '<div class="eda-quiet">Katsayi bilgisi uretilemedi.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.dataframe(coef_df, use_container_width=True, hide_index=True)


def _tab_pii(result: ReportResult, pii_mode: PiiMode) -> None:
    _section("PII Bulgulari")
    findings = result.pii_findings
    findings_df = findings_to_dataframe(findings)
    if findings_df.empty:
        st.info("Kisisel veri (PII) tespit edilmedi.")
        return

    st.dataframe(findings_df, use_container_width=True, hide_index=True)
    st.markdown(
        f'<div class="eda-quiet">{len(findings)} sutun PII olarak isaretlendi.</div>',
        unsafe_allow_html=True,
    )

    if pii_mode != PiiMode.MASK:
        st.info("Maskeleme icin sol menuden PII modunu 'Maske' olarak secin.")
        return

    result = _render_masking_ui(result, findings)


def _render_masked_downloads(df: pd.DataFrame, original_stem: str, wide: bool = True) -> None:
    """Render CSV + Excel download buttons for a masked dataframe."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    xlsx_bytes = buf.getvalue()
    dl1, dl2 = st.columns([1, 1])
    with dl1:
        st.download_button(
            label="Maskeli CSV indir",
            data=csv_bytes,
            file_name=f"masked_{original_stem}.csv",
            mime="text/csv",
            use_container_width=wide,
        )
    with dl2:
        st.download_button(
            label="Maskeli Excel indir",
            data=xlsx_bytes,
            file_name=f"masked_{original_stem}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=wide,
        )


def _render_masking_ui(result: ReportResult, findings: list[PiiFinding]) -> ReportResult:
    """Render per-column mask-mode selectors + preview/download.

    Returns the (possibly updated) result; the original cleaned_df is preserved
    in session_state so previews never mutate state.
    """
    original_df = (
        result.unmasked_cleaned_df
        if result.unmasked_cleaned_df is not None
        else result.cleaned_df
    )
    if original_df is None:
        st.markdown(
            '<div class="eda-quiet">Onizleme icin veri bulunamadi.</div>',
            unsafe_allow_html=True,
        )
        return result

    # Preserve original (unmasked) cleaned_df once.
    if "pii_original_df" not in st.session_state:
        st.session_state["pii_original_df"] = original_df.copy()
    source_df = st.session_state["pii_original_df"]

    _section("Sutun Bazli Maskeleme Modu")
    st.markdown(
        '<div class="eda-quiet">Her PII sutunu icin ayri maskeleme modu secin.</div>',
        unsafe_allow_html=True,
    )
    global_default = st.session_state.get("global_mask_mode", MaskMode.PARTIAL)
    show_last = st.session_state.get("global_show_last", 4)

    column_modes: dict[str, MaskMode] = {}
    for i, finding in enumerate(findings):
        col = finding.column
        default_label = _MASK_LABEL_OF.get(
            st.session_state.get(f"mask_mode_{col}", global_default),
            "Kısmi",
        )
        chosen = st.selectbox(
            f"{col} ({finding.pii_type.value})",
            _MASK_LABELS,
            index=_MASK_LABELS.index(default_label),
            key=f"mask_mode_select_{col}_{i}",
        )
        column_modes[col] = _MASK_MAP[chosen]

    preview_clicked = st.button("Maskeleme Onizle", use_container_width=True)

    if preview_clicked:
        preview_masked = apply_per_column_mask(
            source_df, findings, column_modes, show_last=show_last,
        )
        preview_cols = [f.column for f in findings if f.column in source_df.columns]
        orig_view = source_df[preview_cols].head(50)
        masked_view = preview_masked[preview_cols].head(50)
        st.markdown(
            '<div class="eda-section">Onizleme (ilk 50 satir)</div>',
            unsafe_allow_html=True,
        )
        lo, hi = st.columns([1, 1])
        with lo:
            st.caption("Orijinal")
            st.dataframe(orig_view, use_container_width=True, hide_index=True)
        with hi:
            st.caption("Maskeli")
            st.dataframe(masked_view, use_container_width=True, hide_index=True)

    _section("Maskele ve Indir")
    if st.button("Maskeyi Uygula", type="primary", use_container_width=True):
        final_masked = apply_per_column_mask(
            source_df, findings, column_modes, show_last=show_last,
        )
        st.session_state["pii_final_masked_df"] = final_masked
        st.success("Maskeleme uygulandi. Asagidan indirebilirsiniz.")

    final_df = st.session_state.get("pii_final_masked_df")
    if final_df is not None:
        original_stem = Path(st.session_state.get("original_filename", "veri")).stem
        _render_masked_downloads(final_df, original_stem, wide=True)

    return result


def _tab_reports(result: ReportResult) -> None:
    _section("Raporlar")
    reports = result.reports or {}
    if not reports:
        st.markdown(
            '<div class="eda-quiet">Uretilen rapor bulunamadi.</div>',
            unsafe_allow_html=True,
        )
    for fmt, path in reports.items():
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            st.warning(f"Rapor dosyasi okunamadi: {path}")
            continue
        st.download_button(
            label=f"Indir {fmt.upper()}",
            data=data,
            file_name=Path(path).name,
            use_container_width=False,
        )

    final_masked = st.session_state.get("pii_final_masked_df")
    if final_masked is not None:
        _section("Maskeli Veri (PII tab)")
        original_stem = Path(st.session_state.get("original_filename", "veri")).stem
        _render_masked_downloads(final_masked, original_stem, wide=False)


# ── Pipeline execution ───────────────────────────────────────────────────────


def _run_pipeline(cfg: dict) -> None:
    """Write upload to a temp file, run the pipeline, store result in state."""
    uploaded = cfg["uploaded"]
    temp_path: str | None = None
    try:
        suffix = Path(uploaded.name).suffix or ".csv"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            temp_path = tmp.name

        config = PipelineConfig(
            cleaning=cfg["cleaning_level"],
            pii=cfg["pii_mode"],
            mask_mode=cfg["mask_mode"],
            show_last=cfg["show_last"],
            report_formats=["html", "excel", "pdf", "docx", "pptx"],
        )
        with st.spinner("Analiz Yapiliyor..."):
            result = pipeline.run(temp_path, config)

        # Preserve global mask defaults so the PII tab can reset to them.
        st.session_state["global_mask_mode"] = cfg["mask_mode"]
        st.session_state["global_show_last"] = cfg["show_last"]
        st.session_state["original_filename"] = uploaded.name
        st.session_state["result"] = result
        # Clear stale preview state from a previous run.
        for key in ("pii_original_df", "pii_final_masked_df"):
            st.session_state.pop(key, None)
        st.success("Analiz tamamlandi. Sonuclari sekmelerden inceleyin.")
    except Exception as e:  # noqa: BLE001 - bank-internal users get no stack trace
        st.error(f"Analiz hatasi: {e}")
        st.stop()
    finally:
        if temp_path is not None:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass


# ── Main entrypoint ───────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Veri Analiz Paneli",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_THEME_CSS, unsafe_allow_html=True)
    _render_header()

    cfg = _render_sidebar()

    if cfg["run_clicked"] and cfg["uploaded"] is not None:
        _run_pipeline(cfg)

    result: ReportResult | None = st.session_state.get("result")

    if result is None:
        if cfg["run_clicked"] and cfg["uploaded"] is None:
            st.warning("Lutfen once bir veri dosyasi yukleyin.")
        st.info("Baslamak icin sol menuden bir dosya yukleyip 'Analiz Et'e tiklayin.")
        return

    tab_ozet, tab_desc, tab_gor, tab_ileri, tab_tahmin, tab_model, tab_pii, tab_rapor = st.tabs(
        [
            "Ozet",
            "Tanimlayici Istatistik",
            "Gorseller",
            "Ileri Analiz",
            "Tahmin",
            "Modelleme",
            "PII Uyarilari",
            "Raporlar",
        ]
    )

    original_name = st.session_state.get("original_filename", Path(result.input_path).name)

    with tab_ozet:
        _tab_summary(result, original_name)
    with tab_desc:
        _tab_descriptive(result)
    with tab_gor:
        _tab_charts(result)
    with tab_ileri:
        _tab_advanced(result)
    with tab_tahmin:
        _tab_forecast(result)
    with tab_model:
        _tab_modeling(result)
    with tab_pii:
        _tab_pii(result, cfg["pii_mode"])
    with tab_rapor:
        _tab_reports(result)


if __name__ == "__main__":
    main()
