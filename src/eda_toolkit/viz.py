"""Shared Plotly figure builders for HTML reports and Streamlit dashboard.

Every function returns a plotly.graph_objects.Figure (or None for the
missing-bar helper when the input is empty/invalid). The chart logic is
extracted verbatim from html_reporter.py to keep visual output identical.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def fig_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Build a correlation heatmap (RdBu_r colorscale, z clipped to [-1,1]).

    Parameters
    ----------
    corr : pd.DataFrame
        Square correlation matrix (index and columns are variable names).

    Returns
    -------
    go.Figure
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            texttemplate="%{z:.2f}",
        )
    )
    fig.update_layout(
        title="Korelasyon Matrisi",
        template="plotly_white",
        width=700,
        height=600,
    )
    return fig


def fig_histogram(series: pd.Series, col_name: str) -> go.Figure:
    """Build a histogram for a single numeric column (NA already dropped).

    Parameters
    ----------
    series : pd.Series
        Drop-NA'd numeric series.
    col_name : str
        Column name used for the chart title and x-axis label.

    Returns
    -------
    go.Figure
    """
    fig = px.histogram(series, x=col_name, nbins=30, title=f"{col_name} - Histogram")
    fig.update_layout(template="plotly_white")
    return fig


def fig_cluster_scatter(
    plot_df_valid: pd.DataFrame,
    col_x: str,
    col_y: str,
    fallback_plot_df: pd.DataFrame | None = None,
) -> go.Figure:
    """Build a scatter plot colored by cluster label.

    Uses *plot_df_valid* if non-empty; otherwise falls back to
    *fallback_plot_df*.

    Parameters
    ----------
    plot_df_valid : pd.DataFrame
        DataFrame filtered to exclude unclustered points (label != -1).
    col_x : str
        Name of the column for x-axis.
    col_y : str
        Name of the column for y-axis.
    fallback_plot_df : pd.DataFrame or None
        Full DataFrame (including unclustered) used when *plot_df_valid*
        is empty.

    Returns
    -------
    go.Figure
    """
    title = f"Kumeleme ({col_x} vs {col_y})"
    if len(plot_df_valid) > 0:
        fig = px.scatter(
            plot_df_valid,
            x=col_x,
            y=col_y,
            color="_cluster",
            title=title,
            color_continuous_scale="Viridis",
        )
    else:
        fig = px.scatter(
            fallback_plot_df,
            x=col_x,
            y=col_y,
            color="_cluster",
            title=title,
            color_continuous_scale="Viridis",
        )
    fig.update_layout(template="plotly_white")
    return fig


def fig_cluster_bar(cluster_labels: pd.Series) -> go.Figure:
    """Build a bar chart of cluster distribution (value counts sorted by index).

    Parameters
    ----------
    cluster_labels : pd.Series
        Cluster labels assigned to each row.

    Returns
    -------
    go.Figure
    """
    val_counts = cluster_labels.value_counts().sort_index()
    fig = px.bar(
        x=val_counts.index.astype(str),
        y=val_counts.values,
        title="Kumeleme - Kume Dagilimi",
        labels={"x": "Kume", "y": "Satir Sayisi"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def fig_missing_bar(missing_summary: pd.DataFrame) -> go.Figure | None:
    """Build a horizontal bar chart of missing rates per column.

    Parameters
    ----------
    missing_summary : pd.DataFrame
        DataFrame with columns ``column``, ``missing_count``, ``missing_rate``
        (as produced by ``run_basic_eda``).

    Returns
    -------
    go.Figure or None
        None if the DataFrame is empty or lacks the expected columns, or if
        no column has a positive missing rate.
    """
    if missing_summary is None or missing_summary.empty:
        return None
    if "column" not in missing_summary.columns or "missing_rate" not in missing_summary.columns:
        return None

    # Keep only columns with positive missing rate; sort ascending for a
    # horizontal bar chart so the largest bars are at the top.
    positive = missing_summary[missing_summary["missing_rate"] > 0].copy()
    if positive.empty:
        return None
    positive = positive.sort_values("missing_rate", ascending=True)

    fig = px.bar(
        positive,
        y="column",
        x="missing_rate",
        orientation="h",
        title="Eksik Veri Orani",
        labels={"column": "Sutun", "missing_rate": "Eksik Orani (%)"},
        template="plotly_white",
    )
    return fig
