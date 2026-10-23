from __future__ import annotations

import pandas as pd

from eda_toolkit.types import EdaResult
from eda_toolkit.loader import detect_schema


def run_basic_eda(df: pd.DataFrame) -> EdaResult:
    """Run basic exploratory data analysis.

    Returns EdaResult with:
    - descriptive: numeric columns describe().T
    - missing_summary: column-level missing counts and rates
    - schemas: column classification from loader.detect_schema
    """
    # Descriptive statistics for numeric columns
    numeric_cols = df.select_dtypes(include=["number"])
    if len(numeric_cols.columns) > 0:
        descriptive = numeric_cols.describe().T
    else:
        descriptive = pd.DataFrame()

    # Missing summary
    missing_counts = df.isna().sum()
    missing_rate = (df.isna().sum() / len(df) * 100).round(2)
    missing_summary = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_counts.values,
        "missing_rate": missing_rate.values,
    }).reset_index(drop=True)

    # Schema detection
    schemas = detect_schema(df)

    return EdaResult(
        descriptive=descriptive,
        missing_summary=missing_summary,
        schemas=schemas,
    )
