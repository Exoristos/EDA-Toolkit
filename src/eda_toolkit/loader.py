from __future__ import annotations

from pathlib import Path
from io import BytesIO
import warnings

import pandas as pd
import numpy as np
import requests

from eda_toolkit.types import ColumnSchema


def load_dataframe(path: str) -> pd.DataFrame:
    """Load a CSV or Excel file into a pandas DataFrame.

    CSV files are loaded with pandas.read_csv.
    Excel files (.xlsx, .xls) are loaded via read_excel using openpyxl.

    High-cardinality integer-like columns (IDs such as TCKN) are cast to string
    so they are not treated as numeric measures in EDA (correlation/clustering)
    and remain detectable as PII even when pandas upcast them to float64 due to
    missing values.
    """
    fp = Path(path)
    suffix = fp.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .xlsx/.xls")
    return _cast_id_columns(df)


def _cast_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast high-cardinality integer-like columns to string (in place + return)."""
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        non_null = series.dropna()
        if len(non_null) == 0:
            continue
        unique_count = int(non_null.nunique())
        if _looks_like_id(non_null, unique_count):
            df[col] = series.apply(
                lambda v: str(int(v))
                if pd.notna(v) and float(v).is_integer()
                else (v if pd.isna(v) else str(v))
            )
    return df


def detect_schema(df: pd.DataFrame) -> list[ColumnSchema]:
    """Classify each column and return a list of ColumnSchema entries.

    kind is one of: numeric, categorical, datetime, text, boolean.
    - boolean: 2 unique values OR bool dtype.
    - datetime: columns that parse as datetime.
    - numeric: int/float dtypes.
    - categorical: object dtype with low unique ratio or few uniques (<= 25).
    - text: object dtype with many uniques (> 25 and > 50% unique ratio).
    """
    schemas: list[ColumnSchema] = []
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)
        missing_count = int(series.isna().sum())
        # Drop NaN for unique count on non-null values
        non_null = series.dropna()
        if len(non_null) == 0:
            unique_count = 0
        else:
            unique_count = int(non_null.nunique())

        # Determine kind
        kind = _classify_column(series, n_rows)

        schemas.append(ColumnSchema(
            name=str(col),
            dtype=dtype_str,
            kind=kind,
            unique_count=unique_count,
            missing_count=missing_count,
        ))

    return schemas


def _classify_column(series: pd.Series, n_rows: int) -> str:
    """Classify a single column's kind."""
    # Boolean
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    non_null = series.dropna()
    if len(non_null) == 0:
        return "categorical"  # all missing — treat as categorical
    unique_count = non_null.nunique()

    # 2 uniques => boolean-like
    if unique_count <= 2:
        return "boolean"

    # Datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_object_dtype(series):
        # Try parsing as datetime
        sample = non_null.head(min(20, len(non_null)))
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().sum() >= len(sample) * 0.8:
                return "datetime"
        except (ValueError, TypeError):
            pass

    # Numeric — but check if it looks like an ID (all integers, high cardinality)
    if pd.api.types.is_numeric_dtype(series):
        if _looks_like_id(non_null, unique_count):
            return "text"
        return "numeric"

    # Object / string — classify as categorical or text
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        unique_ratio = unique_count / len(non_null) if len(non_null) > 0 else 0
        if unique_count <= 25 or unique_ratio <= 0.5:
            return "categorical"
        return "text"

    return "text"


def _looks_like_id(series: pd.Series, unique_count: int) -> bool:
    """Check if a numeric series looks like an ID column.

    Returns True if all non-null values are whole numbers (no decimal part),
    the column has high cardinality (> 25 uniques), AND the unique ratio
    is very high (> 0.8), suggesting it is an identifier rather than a
    measurement or categorical-integer.
    """
    n = len(series)
    if n == 0 or unique_count <= 25:
        return False
    unique_ratio = unique_count / n
    if unique_ratio <= 0.8:
        return False
    # Check if all non-null values are effectively integers
    try:
        as_int = series.astype("Int64")
        return bool((as_int.notna() == series.notna()).all())
    except (ValueError, TypeError, pd.errors.IntCastingNaNError):
        return False


def load_many(
    paths: list[str],
    mode: str = "concat",
    on: str | list[str] | None = None,
) -> pd.DataFrame:
    """Load multiple data files and merge them.

    Parameters
    ----------
    paths : list[str]
        List of file paths to load.
    mode : str
        "concat" for vertical concatenation, "join" for keyed merge.
    on : str or list of str, optional
        Key column(s) for join mode. Required when mode="join".

    Returns
    -------
    pd.DataFrame
        Combined DataFrame.

    Raises
    ------
    ValueError
        If paths is empty, or mode is invalid, or join key is missing.
    """
    if not paths:
        raise ValueError("Path listesi bos olamaz")

    if mode not in ("concat", "join"):
        raise ValueError(f"Gecersiz mod: {mode}. 'concat' veya 'join' kullanin")

    # Load all DataFrames
    dataframes: list[pd.DataFrame] = []
    for p in paths:
        dataframes.append(load_dataframe(p))

    if mode == "concat":
        return _concat_dataframes(dataframes, paths)
    else:  # mode == "join"
        return _join_dataframes(dataframes, paths, on)


def _concat_dataframes(
    dataframes: list[pd.DataFrame], paths: list[str]
) -> pd.DataFrame:
    """Concatenate DataFrames vertically with outer union alignment."""
    # Report column-count mismatches via warnings
    if len(dataframes) > 1:
        col_count_ref = len(dataframes[0].columns)
        for df, p in zip(dataframes, paths):
            n = len(df.columns)
            if n != col_count_ref:
                warnings.warn(f"Dosya {Path(p).name} sutun sayisi farkli: {n}")

    result = pd.concat(dataframes, axis=0, join="outer", ignore_index=True)
    return result


def _join_dataframes(
    dataframes: list[pd.DataFrame],
    paths: list[str],
    on: str | list[str] | None,
) -> pd.DataFrame:
    """Merge DataFrames on key(s) with left join, first file as base."""
    if on is None:
        raise ValueError("Join modu icin 'on' parametresi zorunludur")

    # Ensure on is a list for uniform handling
    on_keys = [on] if isinstance(on, str) else list(on)

    # Validate on keys exist in all DataFrames
    for df, p in zip(dataframes, paths):
        missing = [k for k in on_keys if k not in df.columns]
        if missing:
            raise ValueError(
                f"Join anahtari '{missing[0]}' tum dosyalarda yok"
            )

    # Successive left joins
    result = dataframes[0]
    for idx, df in enumerate(dataframes[1:], start=1):
        result = pd.merge(result, df, on=on_keys, how="left")

    return result


def load_from_url(
    url: str,
    fmt: str = "csv",
    headers: dict | None = None,
    timeout: int = 30,
    verify: bool = True,
) -> pd.DataFrame:
    """Load a DataFrame from an HTTP/HTTPS URL.

    Parameters
    ----------
    url : str
        URL to fetch data from.
    fmt : str
        Format of the remote data: "csv", "json", or "xlsx".
    headers : dict, optional
        HTTP headers to send with the request.
    timeout : int
        Request timeout in seconds.
    verify : bool
        Whether to verify the server's TLS certificate.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    ValueError
        If fmt is not one of "csv", "json", "xlsx".
    RuntimeError
        If the HTTP request fails or times out.
    """
    try:
        response = requests.get(
            url, headers=headers, timeout=timeout, verify=verify
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Zaman asimi: {url}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"URL hatasi: {url} - {e}")

    if response.status_code != 200:
        raise RuntimeError(f"URL hatasi {response.status_code}: {url}")

    content = response.content
    buffer = BytesIO(content)

    fmt_lower = fmt.lower()
    if fmt_lower == "csv":
        return pd.read_csv(buffer)
    elif fmt_lower == "json":
        return pd.read_json(buffer)
    elif fmt_lower in ("xlsx", "xls"):
        return pd.read_excel(buffer, engine="openpyxl")
    else:
        raise ValueError(
            f"Desteklenmeyen format: {fmt}. 'csv', 'json' veya 'xlsx' kullanin"
        )
