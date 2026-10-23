"""Static matplotlib PNG builders for Word/PPTX bank reports.

Mirrors the Plotly figures in viz.py with identical titles, colorscales,
and flavour. Uses the Agg backend; every function returns PNG bytes (or None
when input is empty/invalid). No plt.show() — designed for embedding.
"""

from __future__ import annotations

import io
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


# ── constants ────────────────────────────────────────────────────────────────

DPI = 110
FIG_W = 7.0
FIG_H = 5.5


def _fig_to_png(fig: plt.Figure) -> bytes:
    """Render a figure to PNG bytes via BytesIO, then close."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.2)
    buf.seek(0)
    data = buf.getvalue()
    buf.close()
    plt.close(fig)
    return data


# ── public builders ──────────────────────────────────────────────────────────


def png_correlation_heatmap(corr: pd.DataFrame) -> bytes | None:
    """RdBu_r heatmap, vmin=-1 / vmax=1, title "Korelasyon Matrisi".

    Returns PNG bytes, or None when corr is empty/invalid.
    """
    if corr is None or corr.empty:
        return None

    labels = list(corr.columns)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(FIG_W, n * 1.0), max(FIG_H, n * 0.9)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Korelasyon Matrisi", fontsize=12, fontweight="bold")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Korelasyon", fontsize=8)

    fig.tight_layout()
    return _fig_to_png(fig)


def png_histogram(series: pd.Series, col_name: str) -> bytes | None:
    """30-bin histogram with title f"{col_name} - Histogram".

    Returns PNG bytes, or None when series is empty.
    """
    if series is None or series.empty or series.dropna().empty:
        return None

    clean = series.dropna()
    if len(clean) == 0:
        return None

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.hist(clean.values, bins=30, color="#1E3A5F", edgecolor="white", alpha=0.85)
    ax.set_title(f"{col_name} - Histogram", fontsize=12, fontweight="bold")
    ax.set_xlabel(col_name, fontsize=9)
    ax.set_ylabel("Frekans", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return _fig_to_png(fig)


def png_missing_bar(missing_summary: pd.DataFrame) -> bytes | None:
    """Horizontal bar chart of missing rate per column, title "Eksik Veri Orani".

    Returns None when the summary is empty or has no positive missing rates.
    """
    if missing_summary is None or missing_summary.empty:
        return None
    if "column" not in missing_summary.columns or "missing_rate" not in missing_summary.columns:
        return None

    positive = missing_summary[missing_summary["missing_rate"] > 0].copy()
    if positive.empty:
        return None

    positive = positive.sort_values("missing_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(FIG_W, max(3.0, len(positive) * 0.4)))
    y_pos = range(len(positive))
    ax.barh(y_pos, positive["missing_rate"].values, color="#1E3A5F", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(positive["column"].values, fontsize=8)
    ax.set_xlabel("Eksik Orani (%)", fontsize=9)
    ax.set_title("Eksik Veri Orani", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return _fig_to_png(fig)


def png_cluster_scatter(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    cluster_labels: pd.Series | None = None,
) -> bytes | None:
    """Viridis scatter plot colored by cluster, title f"Kumeleme ({col_x} vs {col_y})".

    Returns PNG bytes, or None when df is empty.
    """
    if df is None or df.empty:
        return None

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    if cluster_labels is not None and len(cluster_labels) == len(df):
        labels = cluster_labels.values
    elif "_cluster" in df.columns:
        labels = df["_cluster"].values
    else:
        labels = np.zeros(len(df))

    scatter = ax.scatter(
        df[col_x].values,
        df[col_y].values,
        c=labels,
        cmap="viridis",
        alpha=0.7,
        s=20,
        edgecolors="none",
    )
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Kume", fontsize=8)

    ax.set_xlabel(col_x, fontsize=9)
    ax.set_ylabel(col_y, fontsize=9)
    ax.set_title(f"Kumeleme ({col_x} vs {col_y})", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return _fig_to_png(fig)
