from __future__ import annotations

import pandas as pd
import numpy as np

from eda_toolkit.types import CleaningLevel, CleaningAction

# Missing-value markers that should be treated as NaN (case-insensitive).
_MISSING_MARKERS: set[str] = {"", "na", "n/a", "-", "null", "yok"}

# Boolean mapping for Turkish and English boolean-like values.
_BOOL_MAP: dict[str, bool] = {
    "evet": True,
    "hayır": False,
    "hayir": False,
    "e": True,
    "h": False,
    "true": True,
    "false": False,
    "1": True,
    "0": False,
}


def _is_string_like(series: pd.Series) -> bool:
    """Return True for object or StringDtype columns (pandas 2 and 3 compat)."""
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(
        series
    )


def clean(
    df: pd.DataFrame, level: CleaningLevel
) -> tuple[pd.DataFrame, list[CleaningAction]]:
    """Clean the dataframe at the given level.

    CONSERVATIVE: no-op, returns unchanged copy and empty log.
    MODERATE: safe, non-destructive fixes only (whitespace, missing markers, type
              coercion where unambiguous). Does NOT impute or remove outliers.
    AGGRESSIVE: everything moderate does, plus missing-value imputation (median
                for numeric, mode for categorical) and IQR-based outlier capping
                (winsorize — rows are preserved).
    """
    df = df.copy()
    cleaning_log: list[CleaningAction] = []

    if level == CleaningLevel.CONSERVATIVE:
        return (df, cleaning_log)

    _apply_moderate_fixes(df, cleaning_log)

    if level == CleaningLevel.AGGRESSIVE:
        _impute_missing(df, cleaning_log)
        _cap_outliers(df, cleaning_log)

    return (df, cleaning_log)


# ── MODERATE helpers ──────────────────────────────────────────────────────────


def _apply_moderate_fixes(df: pd.DataFrame, log: list[CleaningAction]) -> None:
    """Apply safe, non-destructive-to-values fixes at the MODERATE level."""
    _trim_whitespace(df, log)
    _normalize_missing_markers(df, log)
    _coerce_types(df, log)


def _trim_whitespace(df: pd.DataFrame, log: list[CleaningAction]) -> None:
    """Trim leading/trailing whitespace from string cells."""
    for col in df.columns:
        col_values = df[col]
        if not _is_string_like(col_values):
            continue
        # Operate on cells that are actually strings (skip NA/NaN)
        mask = col_values.notna() & col_values.apply(lambda x: isinstance(x, str))
        if not mask.any():
            continue
        trimmed = col_values[mask].str.strip()
        changed_mask = trimmed != col_values[mask]
        count = changed_mask.sum()
        if count > 0:
            df.loc[mask & changed_mask, col] = trimmed[changed_mask]
            log.append(
                CleaningAction(
                    column=col,
                    issue="bosluk",
                    level=CleaningLevel.MODERATE,
                    applied=True,
                    detail=f"Baştaki/sondaki boşluklar temizlendi ({count} hücre).",
                )
            )


def _normalize_missing_markers(
    df: pd.DataFrame, log: list[CleaningAction]
) -> None:
    """Convert missing-value marker strings (NA, N/A, -, null, yok, …) to NaN."""
    for col in df.columns:
        col_values = df[col]
        if not _is_string_like(col_values):
            continue
        # Only check non-null, actual-string cells
        str_mask = col_values.notna() & col_values.apply(
            lambda x: isinstance(x, str)
        )
        if not str_mask.any():
            continue
        # Which string cells match a missing marker (case-insensitive, stripped)
        lower_vals = col_values[str_mask].str.strip().str.lower()
        marker_mask = lower_vals.isin(_MISSING_MARKERS)
        count = marker_mask.sum()
        if count > 0:
            idx = col_values[str_mask].index[marker_mask]
            df.loc[idx, col] = np.nan
            log.append(
                CleaningAction(
                    column=col,
                    issue="eksik_isaretci",
                    level=CleaningLevel.MODERATE,
                    applied=True,
                    detail=(
                        f"Eksik değer işaretçileri NaN yapıldı "
                        f"({count} hücre)."
                    ),
                )
            )


def _coerce_types(df: pd.DataFrame, log: list[CleaningAction]) -> None:
    """Best-effort type coercion for string columns.

    Numeric-looking strings → float/int.
    Boolean-like Turkish/English strings → bool.
    Does NOT force coercion when ambiguous.
    """
    for col in df.columns:
        col_values = df[col]
        if not _is_string_like(col_values):
            continue

        # Skip fully-NaN / completely empty columns
        if col_values.dropna().empty:
            continue

        # --- Boolean coercion (try first — bool values are unambiguous) ---
        bool_coerced = _try_coerce_bool(col_values)
        if bool_coerced is not None:
            df[col] = bool_coerced
            log.append(
                CleaningAction(
                    column=col,
                    issue="tur_bool",
                    level=CleaningLevel.MODERATE,
                    applied=True,
                    detail="Mantıksal (evet/hayır) değerler bool türe dönüştürüldü.",
                )
            )
            continue  # Don't also try numeric for bool columns

        # --- Numeric coercion ---
        numeric_coerced = _try_coerce_numeric(col_values)
        if numeric_coerced is not None:
            df[col] = numeric_coerced
            log.append(
                CleaningAction(
                    column=col,
                    issue="tur_sayisal",
                    level=CleaningLevel.MODERATE,
                    applied=True,
                    detail="Sayısal görünümlü değerler sayı türüne dönüştürüldü.",
                )
            )


def _try_coerce_bool(series: pd.Series) -> pd.Series | None:
    """Try to coerce a string series to nullable boolean. Returns None if ambiguous."""
    # Only operate on non-null values that are strings
    str_mask = series.notna() & series.apply(lambda x: isinstance(x, str))
    str_vals = series[str_mask]

    if str_vals.empty:
        return None

    # Strip, lower
    cleaned = str_vals.str.strip().str.lower()

    # Check: do all string values match the bool map?
    known = cleaned.isin(_BOOL_MAP)
    if not known.all():
        return None  # Some strings aren't recognized → ambiguous, leave alone

    # Coerce — use nullable boolean dtype (pd.BooleanDtype)
    result = pd.Series(index=series.index, dtype="boolean")
    mapped = cleaned.map(_BOOL_MAP)
    result[str_mask] = mapped
    # NaN / non-string values remain NA in nullable boolean
    return result


def _try_coerce_numeric(series: pd.Series) -> pd.Series | None:
    """Try to coerce a string series to numeric. Returns None if too many failures."""
    str_mask = series.notna() & series.apply(lambda x: isinstance(x, str))
    str_vals = series[str_mask]

    if str_vals.empty:
        return None

    # Attempt numeric conversion
    coerced = pd.to_numeric(str_vals.str.strip(), errors="coerce")

    # If more than 30% of the non-null string values failed conversion, leave it
    failure_rate = coerced.isna().mean()
    if failure_rate > 0.3:
        return None

    # Build result: preserve any existing numeric values, apply coerced ones
    result = pd.Series(index=series.index, dtype="float64")
    # Copy any existing numeric values (int/float, not bool)
    num_mask = series.apply(
        lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
    )
    if num_mask.any():
        result[num_mask] = pd.to_numeric(series[num_mask], errors="coerce")
    # Apply coerced string values
    result[str_mask] = coerced

    # Convert to nullable Int64 if all non-null values are whole numbers
    non_null = result.dropna()
    if len(non_null) > 0 and (non_null == non_null.astype("int64")).all():
        return result.astype("Int64")

    return result


# ── AGGRESSIVE helpers ────────────────────────────────────────────────────────

# Column-level missing-rate threshold above which we log a note but still impute.
_HIGH_MISSING_THRESHOLD = 0.5


def _impute_missing(df: pd.DataFrame, log: list[CleaningAction]) -> None:
    """Impute missing values: numeric columns → median; categorical → mode.

    If a column is >50% missing, impute still but log that fact explicitly.
    """
    for col in df.columns:
        col_values = df[col]
        missing_mask = col_values.isna()
        missing_count = missing_mask.sum()
        if missing_count == 0:
            continue
        missing_rate = missing_count / len(df)

        if missing_rate > _HIGH_MISSING_THRESHOLD:
            log.append(
                CleaningAction(
                    column=col,
                    issue="yuksek_eksik",
                    level=CleaningLevel.AGGRESSIVE,
                    applied=True,
                    detail=(
                        f"Sütunun %{missing_rate*100:.0f}'i eksik; buna rağmen "
                        f"imputasyon uygulandı."
                    ),
                )
            )

        # Boolean columns are "numeric" per pandas but median returns float,
        # which fillna rejects for BooleanArray. Use mode instead.
        if pd.api.types.is_numeric_dtype(col_values) and not pd.api.types.is_bool_dtype(col_values):
            filler = col_values.median()
            method = "medyan"
        else:
            # Categorical / object / string: use mode (most frequent)
            mode_vals = col_values.mode()
            filler = mode_vals.iloc[0] if len(mode_vals) > 0 else "Bilinmiyor"
            method = "mod"

        df[col] = col_values.fillna(filler)
        log.append(
            CleaningAction(
                column=col,
                issue="eksik_doldurma",
                level=CleaningLevel.AGGRESSIVE,
                applied=True,
                detail=(
                    f"Eksik değerler {method} ile dolduruldu "
                    f"({missing_count} değer; {filler})."
                ),
            )
        )


def _cap_outliers(df: pd.DataFrame, log: list[CleaningAction]) -> None:
    """Winsorize (IQR-cap) numeric columns to [Q1−1.5*IQR, Q3+1.5*IQR].

    Capping is preferred over removal to preserve row count.
    Only numeric columns with sufficient non-null values are processed.
    Boolean columns are skipped because pandas treats them as numeric,
    but numpy quantile fails on them.
    """
    for col in df.columns:
        col_values = df[col]
        if not pd.api.types.is_numeric_dtype(col_values):
            continue
        if pd.api.types.is_bool_dtype(col_values):
            continue
        non_null = col_values.dropna()
        if len(non_null) < 4:
            continue  # Not enough data for IQR

        Q1 = non_null.quantile(0.25)
        Q3 = non_null.quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            continue  # No variance — nothing to cap

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Count values outside bounds
        low_mask = col_values < lower
        high_mask = col_values > upper
        capped_count = (low_mask | high_mask).sum()

        if capped_count == 0:
            continue

        # Winsorize: cap low values to lower bound, high values to upper bound
        df.loc[low_mask, col] = lower
        df.loc[high_mask, col] = upper

        log.append(
            CleaningAction(
                column=col,
                issue="aykiri_deger",
                level=CleaningLevel.AGGRESSIVE,
                applied=True,
                detail=(
                    f"Aykırı değerler IQR ile sınırlandırıldı "
                    f"({capped_count} değer; alt={lower:.2f}, üst={upper:.2f})."
                ),
            )
        )


# ── Public helper ─────────────────────────────────────────────────────────────


def summarize_issues(df: pd.DataFrame) -> list[CleaningAction]:
    """Report issues WITHOUT fixing them — useful for conservative-mode previews.

    Returns a list of CleaningAction entries with applied=False describing
    what WOULD be cleaned at higher levels.
    """
    issues: list[CleaningAction] = []
    _report_missing_markers(df, issues)
    _report_type_issues(df, issues)
    _report_missing_values(df, issues)
    _report_outliers(df, issues)
    return issues


def _report_missing_markers(
    df: pd.DataFrame, issues: list[CleaningAction]
) -> None:
    for col in df.columns:
        col_values = df[col]
        if not _is_string_like(col_values):
            continue
        str_mask = col_values.notna() & col_values.apply(
            lambda x: isinstance(x, str)
        )
        if not str_mask.any():
            continue
        lower_vals = col_values[str_mask].str.strip().str.lower()
        count = lower_vals.isin(_MISSING_MARKERS).sum()
        if count > 0:
            issues.append(
                CleaningAction(
                    column=col,
                    issue="eksik_isaretci",
                    level=CleaningLevel.CONSERVATIVE,
                    applied=False,
                    detail=(
                        f"Tespit: {count} eksik değer işaretçisi "
                        f"(NA, boş, -, vb.) bulundu."
                    ),
                )
            )


def _report_type_issues(
    df: pd.DataFrame, issues: list[CleaningAction]
) -> None:
    for col in df.columns:
        col_values = df[col]
        if not _is_string_like(col_values):
            continue
        if col_values.dropna().empty:
            continue
        # Try bool
        if _try_coerce_bool(col_values) is not None:
            issues.append(
                CleaningAction(
                    column=col,
                    issue="tur_bool",
                    level=CleaningLevel.CONSERVATIVE,
                    applied=False,
                    detail=(
                        "Tespit: Sütun mantıksal (evet/hayır) değerler "
                        "içeriyor — bool dönüşümü mümkün."
                    ),
                )
            )
            continue
        # Try numeric
        if _try_coerce_numeric(col_values) is not None:
            issues.append(
                CleaningAction(
                    column=col,
                    issue="tur_sayisal",
                    level=CleaningLevel.CONSERVATIVE,
                    applied=False,
                    detail=(
                        "Tespit: Sütun sayısal görünümlü değerler "
                        "içeriyor — sayı dönüşümü mümkün."
                    ),
                )
            )


def _report_missing_values(
    df: pd.DataFrame, issues: list[CleaningAction]
) -> None:
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            issues.append(
                CleaningAction(
                    column=col,
                    issue="eksik_doldurma",
                    level=CleaningLevel.CONSERVATIVE,
                    applied=False,
                    detail=f"Tespit: {missing} eksik değer — imputasyon mümkün.",
                )
            )


def _report_outliers(
    df: pd.DataFrame, issues: list[CleaningAction]
) -> None:
    for col in df.columns:
        col_values = df[col]
        if not pd.api.types.is_numeric_dtype(col_values):
            continue
        if pd.api.types.is_bool_dtype(col_values):
            continue
        non_null = col_values.dropna()
        if len(non_null) < 4:
            continue
        Q1 = non_null.quantile(0.25)
        Q3 = non_null.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_count = ((col_values < lower) | (col_values > upper)).sum()
        if outlier_count > 0:
            issues.append(
                CleaningAction(
                    column=col,
                    issue="aykiri_deger",
                    level=CleaningLevel.CONSERVATIVE,
                    applied=False,
                    detail=f"Tespit: {outlier_count} aykırı değer (IQR yöntemi).",
                )
            )
