"""Korelasyon matrisi ve kategorik değişken özeti."""

from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Nümerik sütunlar için Pearson korelasyon matrisi.

    Args:
        df: Girdi DataFrame.

    Returns:
        Korelasyon matrisi (indeks/sütun = nümerik kolon adları).
        Hiç nümerik sütun yoksa boş DataFrame döner.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty or numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr(method="pearson")


def categorical_summary(df: pd.DataFrame) -> dict:
    """Kategorik sütunlar için değer sayımları ve frekans bilgisi.

    Yalnızca object veya category dtype sütunlar incelenir.

    Args:
        df: Girdi DataFrame.

    Returns:
        {column: {"counts": {value: count, ...}, "n_unique": int}, ...}
    """
    result: dict = {}
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(df[col]):
            series = df[col].dropna()
            counts = series.value_counts()
            result[col] = {
                "counts": counts.to_dict(),
                "n_unique": int(series.nunique()),
            }
    return result
