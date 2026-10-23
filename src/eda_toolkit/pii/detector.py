"""
Three-layer Turkish PII column detector.

Layer 1 - Column-name heuristics: TR + EN keyword maps for each PiiType.
Layer 2 - Value validation: sample N values, apply regex pre-filter + checksum
           validators; if hit-rate > threshold → PiiFinding.
Layer 3 - Entropy/dtype: tie-breaker / supplement for numeric-like PII columns.

Public API:
    detect_pii(df, sample_size=100, threshold=0.8) -> list[PiiFinding]
"""

from __future__ import annotations

import math
import re
from typing import Callable

import pandas as pd

from eda_toolkit.types import PiiFinding, PiiType
from eda_toolkit.pii.validators import (
    is_valid_tckn,
    is_valid_vkn,
    is_valid_iban,
    is_valid_phone,
    is_valid_email,
    is_valid_card_pan,
    match_rate,
    _strip_non_digits,
    _strip_dialable,
)

# ── Column-name keyword sets (layer 1) ───────────────────────────────────────

_KEYWORDS: dict[PiiType, set[str]] = {
    PiiType.TCKN: {
        "tc_kimlik", "tckn", "tc_kimlik_no", "kimlik_no", "kimlik",
        "t.c.", "national_id", "citizen_id", "identity_number",
        "tc", "tcno", "tckimlikno",
    },
    PiiType.VKN: {
        "vergi_no", "vkn", "vergi_kimlik", "vergi_kimlik_no",
        "tax_id", "tax_number", "vergi",
    },
    PiiType.IBAN: {
        "iban", "hesap_no", "account_number", "iban_no",
        "iban_no", "hesap_numarasi",
    },
    PiiType.PHONE: {
        "telefon", "tel", "cep", "gsm", "phone", "mobile", "cell",
        "tel_no", "telefon_no", "telefon_numarasi", "cep_tel",
    },
    PiiType.EMAIL: {
        "email", "e-posta", "mail", "eposta", "e_mail",
    },
    PiiType.NAME: {
        "ad", "soyad", "isim", "ad_soyad", "name", "first_name",
        "last_name", "surname", "adi", "soyadi", "tam_ad",
    },
    PiiType.DOB: {
        "dogum_tarihi", "dtarih", "d_tarih", "birth_date", "dob",
        "dogum", "dogum_gunu",
    },
    PiiType.ADDRESS: {
        "adres", "acik_adres", "ikamet", "address", "adresi",
        "ev_adresi", "is_adresi",
    },
    PiiType.CARD_PAN: {
        "kart_no", "kart", "pan", "card_number", "cc",
        "kredi_karti", "kart_numarasi", "cc_no",
    },
}

# ── Regex pre-filters (cheap, run before checksum) ───────────────────────────

_RE_TCKN_PREFILTER = re.compile(r"^[1-9]\d{10}$")
_RE_VKN_PREFILTER = re.compile(r"^\d{10}$")
_RE_IBAN_PREFILTER = re.compile(r"^TR\d{24}$", re.IGNORECASE)
# Phone: spec is (+90|0)+10 digits, but datasets often omit the prefix.
# Pre-filter accepts common Turkish mobile patterns (5xx or 0/+90).
_RE_PHONE_PREFILTER = re.compile(r"^(\+90|0)?5\d{9}$")
_RE_PAN_PREFILTER = re.compile(r"^\d{13,19}$")

# Per-type config: (regex pre-filter, checksum validator, min_length_hint)
_TYPE_CONFIG: dict[PiiType, tuple[re.Pattern | None, Callable | None, int | None]] = {
    PiiType.TCKN: (_RE_TCKN_PREFILTER, is_valid_tckn, 11),
    PiiType.VKN: (_RE_VKN_PREFILTER, is_valid_vkn, 10),
    PiiType.IBAN: (_RE_IBAN_PREFILTER, is_valid_iban, 26),
    PiiType.PHONE: (_RE_PHONE_PREFILTER, is_valid_phone, None),
    PiiType.EMAIL: (None, is_valid_email, None),
    PiiType.CARD_PAN: (_RE_PAN_PREFILTER, is_valid_card_pan, None),
    PiiType.NAME: (None, None, None),       # no checksum
    PiiType.DOB: (None, None, None),        # no checksum
    PiiType.ADDRESS: (None, None, None),    # no checksum
}


def _normalize_column_name(name: str) -> str:
    """Lowercase and normalize column name for keyword matching."""
    # Replace common separators with underscore for matching
    n = name.lower().strip()
    n = re.sub(r"[-\s]+", "_", n)
    n = re.sub(r"[^a-z0-9_.]", "", n)
    return n


def _column_name_match(col_name: str, keywords: set[str]) -> float:
    """Return 1.0 if column name contains any keyword, else 0.0."""
    norm = _normalize_column_name(col_name)
    for kw in keywords:
        if kw in norm:
            return 1.0
    return 0.0


def _get_type_for_column(col_name: str) -> list[tuple[PiiType, float]]:
    """Return list of (PiiType, score) from column-name matching.

    Score is 1.0 for exact-ish keyword match, lower otherwise.
    """
    norm = _normalize_column_name(col_name)
    results: list[tuple[PiiType, float]] = []
    for pii_type, keywords in _KEYWORDS.items():
        for kw in keywords:
            if kw == norm:
                results.append((pii_type, 1.0))
                break
            elif kw in norm:
                results.append((pii_type, 0.8))
                break
    return results


def _normalize_value(v) -> str:
    """Normalize a scalar to a clean string for PII regex/checksum matching.

    Integer-valued floats (from float64 upcast due to NaN) become plain digit
    strings: 93755037582.0 -> "93755037582". Non-integer floats and other
    values are stringified as-is. This is defense-in-depth alongside the
    loader's ID-column cast.
    """
    # Numeric integer-valued -> digit string without decimal/exponent
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return str(v)
    return str(v).strip()


def _sample_non_null(series: pd.Series, sample_size: int) -> list:
    """Return up to *sample_size* non-null, non-empty string values from series.

    Values are normalized (integer-valued floats -> digit strings) so that PII
    detection works on columns pandas upcast to float64 due to missing values.
    """
    s = series.dropna()
    s = s.apply(_normalize_value)
    s = s[s != ""]
    if len(s) > sample_size:
        s = s.sample(n=sample_size, random_state=42)
    return s.tolist()


def _entropy_score(series: pd.Series) -> float:
    """Heuristic entropy score for a column.

    High digit ratio + narrow length range → more likely PII numeric.
    Returns score in [0, 1].
    """
    non_null = series.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]
    if len(non_null) == 0:
        return 0.0

    digit_ratio = non_null.apply(lambda s: sum(1 for c in s if c.isdigit()) / max(len(s), 1))
    avg_digit_ratio = digit_ratio.mean()

    # Length consistency: low std → more PII-like
    lengths = non_null.str.len()
    if len(lengths) > 1:
        length_cv = lengths.std() / max(lengths.mean(), 1)  # coefficient of variation
        length_score = max(0.0, 1.0 - length_cv)
    else:
        length_score = 1.0

    return (avg_digit_ratio * 0.6 + length_score * 0.4)


# ── Main detection function ───────────────────────────────────────────────────

def detect_pii(
    df: pd.DataFrame,
    sample_size: int = 100,
    threshold: float = 0.8,
) -> list[PiiFinding]:
    """Detect PII columns in *df* using a three-layer approach.

    Args:
        df: Input DataFrame.
        sample_size: Max values to sample per column for value-layer checks.
        threshold: Minimum hit-rate (validated / sampled) for value-layer detection.

    Returns:
        list[PiiFinding]: Detected PII columns, ordered by confidence descending.
    """
    findings: list[PiiFinding] = []

    for col_name in df.columns:
        series = df[col_name]
        name_matches = _get_type_for_column(col_name)

        # ---- Layer 1: Column-name heuristics ----
        name_pii_types: set[PiiType] = {t for t, _ in name_matches}
        name_scores: dict[PiiType, float] = {t: s for t, s in name_matches}

        # ---- Layer 2: Value validation for types with checksums ----
        sampled = _sample_non_null(series, sample_size)
        sample_count = len(sampled)

        if sample_count < 3:
            # Too few non-null values to make a reliable call
            continue

        # For each PII type that has a validator, check sample values
        for pii_type, (pre_filter, validator, _min_len) in _TYPE_CONFIG.items():
            if validator is None:
                continue  # NAME, DOB, ADDRESS have no checksum

            # Pre-filter: does the sample look like this type?
            if pre_filter is not None:
                # Clean values for pre-filter (strip formatting)
                cleaned_sample: list[str] = []
                for v in sampled:
                    if pii_type == PiiType.PHONE:
                        cleaned_sample.append(_strip_dialable(v))
                    elif pii_type == PiiType.IBAN:
                        cleaned_sample.append(v.replace(" ", "").upper())
                    elif pii_type == PiiType.CARD_PAN:
                        cleaned_sample.append(re.sub(r"[\s\-]", "", v))
                    else:
                        cleaned_sample.append(_strip_non_digits(v))
                pre_pass = sum(1 for v in cleaned_sample if pre_filter.match(v))
                pre_rate = pre_pass / sample_count
                # If very few look like this type, skip expensive checksum
                if pre_rate < 0.2:
                    continue
                # Use cleaned sample for validation
                validate_sample = cleaned_sample
            else:
                validate_sample = sampled
                pre_rate = 1.0

            # Run checksum validator
            validated = sum(1 for v in validate_sample if validator(v))
            hit_rate = validated / sample_count
            name_score = name_scores.get(pii_type, 0.0)

            # Accept if checksum hit-rate >= threshold
            if hit_rate >= threshold:
                if name_score > 0:
                    confidence = min(1.0, name_score * 0.4 + hit_rate * 0.6)
                    det_layer = "name+value"
                else:
                    confidence = hit_rate * 0.85
                    det_layer = "value"

                findings.append(PiiFinding(
                    column=col_name,
                    pii_type=pii_type,
                    confidence=round(confidence, 3),
                    match_rate=round(hit_rate, 3),
                    sample_count=sample_count,
                    detection_layer=det_layer,
                ))
            # Fallback: name match + regex pre-filter match at high rate
            # Useful when checksum data is buggy (e.g., IBANs with wrong MOD-97)
            elif name_score >= 0.8 and pre_rate >= threshold:
                confidence = name_score * 0.5 + pre_rate * 0.35
                findings.append(PiiFinding(
                    column=col_name,
                    pii_type=pii_type,
                    confidence=round(confidence, 3),
                    match_rate=round(pre_rate, 3),
                    sample_count=sample_count,
                    detection_layer="name+pattern",
                ))

        # ---- Layer 2: Name-only types (NAME, DOB, ADDRESS) ----
        for pii_type in (PiiType.NAME, PiiType.DOB, PiiType.ADDRESS):
            name_score = name_scores.get(pii_type, 0.0)
            if name_score > 0:
                # Lower confidence for name-only detection
                confidence = name_score * 0.6
                findings.append(PiiFinding(
                    column=col_name,
                    pii_type=pii_type,
                    confidence=round(confidence, 3),
                    match_rate=1.0,  # Name match is binary
                    sample_count=sample_count,
                    detection_layer="name",
                ))

        # ---- Layer 3: Entropy/dtype supplement ----
        # Only for numeric-like columns that weren't caught by name
        if not any(f.column == col_name for f in findings):
            ent = _entropy_score(series)
            dtype = str(series.dtype)
            # Check if high-digit-ratio numeric column might be PII
            if ent > 0.85:
                # Could be TCKN/VKN/PAN that wasn't caught by name
                # Try regex + checksum anyway (in case pre-filter was too strict)
                digits_clean = [_strip_non_digits(v) for v in sampled]
                # TCKN-like
                tckn_pre = sum(1 for v in digits_clean if _RE_TCKN_PREFILTER.match(v))
                if tckn_pre / sample_count >= threshold:
                    tckn_valid = sum(1 for v in digits_clean if is_valid_tckn(v))
                    rate = tckn_valid / sample_count
                    if rate >= threshold:
                        findings.append(PiiFinding(
                            column=col_name, pii_type=PiiType.TCKN,
                            confidence=round(rate * 0.8, 3),
                            match_rate=round(rate, 3),
                            sample_count=sample_count,
                            detection_layer="entropy",
                        ))
                        continue

                # VKN-like
                vkn_pre = sum(1 for v in digits_clean if _RE_VKN_PREFILTER.match(v))
                if vkn_pre / sample_count >= threshold:
                    vkn_valid = sum(1 for v in digits_clean if is_valid_vkn(v))
                    rate = vkn_valid / sample_count
                    if rate >= threshold:
                        findings.append(PiiFinding(
                            column=col_name, pii_type=PiiType.VKN,
                            confidence=round(rate * 0.8, 3),
                            match_rate=round(rate, 3),
                            sample_count=sample_count,
                            detection_layer="entropy",
                        ))
                        continue

                # PAN-like
                pan_clean = [re.sub(r"[\s\-]", "", v) for v in sampled]
                pan_pre = sum(1 for v in pan_clean if _RE_PAN_PREFILTER.match(v))
                if pan_pre / sample_count >= threshold:
                    pan_valid = sum(1 for v in pan_clean if is_valid_card_pan(v))
                    rate = pan_valid / sample_count
                    if rate >= threshold:
                        findings.append(PiiFinding(
                            column=col_name, pii_type=PiiType.CARD_PAN,
                            confidence=round(rate * 0.8, 3),
                            match_rate=round(rate, 3),
                            sample_count=sample_count,
                            detection_layer="entropy",
                        ))
                        continue

    # Sort by confidence descending
    findings.sort(key=lambda f: f.confidence, reverse=True)
    return findings
