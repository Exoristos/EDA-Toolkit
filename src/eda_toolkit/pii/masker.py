"""
PII masking module.

Modes (MaskMode):
  - partial: Show last N characters, rest replaced with '*'.
  - full: Replace entire value with "[REDACTED]".
  - hash: SHA-256( salt + value ) truncated hex — deterministic, irreversible.
  - tokenize: Deterministic hash-based token (short ID).  NOT reversable
    without a vault; for MVP this is a deterministic hash prefix.  Documented
    as non-reversible.
  - pseudonymize: Generate a fake-but-valid value of the same PII type
    (valid TCKN via checksum gen, valid IBAN via MOD-97 gen, random
    phone/email, random Turkish name).

Public API:
    mask_value(value, pii_type, mode, show_last=4, salt=None) -> str | None
    mask_dataframe(df, findings, mode, show_last=4, salt=None) -> pd.DataFrame
"""

from __future__ import annotations

import hashlib
import random
import re
import string
from typing import Any

import pandas as pd

from eda_toolkit.types import MaskMode, PiiFinding, PiiType

# ── Generators for pseudonymization ───────────────────────────────────────────

# Turkish first names (common)
_TR_FIRST_NAMES = [
    "Ali", "Mehmet", "Ahmet", "Mustafa", "Hasan", "Huseyin", "Ibrahim",
    "Osman", "Yusuf", "Murat", "Kemal", "Omer", "Fatih", "Emre", "Burak",
    "Ayse", "Fatma", "Emine", "Zeynep", "Hatice", "Elif", "Meryem",
    "Sibel", "Derya", "Gul", "Sevgi", "Nur", "Canan", "Pinar", "Deniz",
]

_TR_LAST_NAMES = [
    "Yilmaz", "Kaya", "Demir", "Celik", "Sahin", "Yildiz", "Ozdemir",
    "Aydin", "Arslan", "Dogan", "Kilic", "Aslan", "Cetin", "Kara", "Koc",
]

_TR_CITIES = [
    "Istanbul", "Ankara", "Izmir", "Bursa", "Antalya", "Adana", "Konya",
    "Gaziantep", "Mersin", "Kayseri", "Eskisehir", "Samsun", "Diyarbakir",
    "Trabzon", "Erzurum", "Kocaeli", "Sakarya", "Malatya", "Manisa", "Balikesir",
]

_TR_STREETS = [
    "Ataturk Cd.", "Cumhuriyet Cd.", "Bagdat Cd.", "Istiklal Cd.",
    "Inonu Cd.", "Gazi Cd.", "Fatih Cd.", "Zafer Cd.",
]

_EMAIL_DOMAINS = [
    "example.com", "anonymized.local", "masked.org", "pseudo.net", "fake.test",
]


def _random_tckn() -> str:
    """Generate a random valid TCKN."""
    digits = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(8)]
    A = sum(digits[i] for i in range(0, 9, 2))
    B = sum(digits[i] for i in range(1, 8, 2))
    d10 = (A * 7 - B) % 10
    d11 = (sum(digits) + d10) % 10
    return "".join(str(d) for d in digits + [d10, d11])


def _random_vkn() -> str:
    """Generate a random valid VKN (authoritative GİB algorithm, matches stdnum)."""
    digits = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(8)]
    s = 0
    for i, n in enumerate(reversed(digits), 1):
        c1 = (n + i) % 10
        if c1:
            c2 = (c1 * (2 ** i)) % 9 or 9
            s += c2
    check = (10 - s) % 10
    return "".join(str(d) for d in digits + [check])


def _random_iban() -> str:
    """Generate a random valid TR IBAN.

    Standard MOD-97 IBAN check digit: rearrange bban + TR00 to end,
    convert to numeric, remainder = numeric % 97, check = 98 - remainder.
    """
    bank_code = f"{random.randint(0, 99999):05d}"
    reserved = "0"
    account = f"{random.randint(0, 9999999999999999):016d}"
    bban = bank_code + reserved + account

    # Rearrange: bban + TR00 (per IBAN standard)
    rearranged = bban + "TR00"
    numeric_str = ""
    for ch in rearranged:
        if ch.isalpha():
            numeric_str += str(ord(ch) - 55)
        else:
            numeric_str += ch
    remainder = int(numeric_str) % 97
    check = 98 - remainder
    return f"TR{check:02d}{bban}"


def _random_phone() -> str:
    """Generate a random Turkish phone (0 + 10 digits)."""
    return f"05{random.randint(10, 99):02d}{random.randint(0, 9999999):07d}"


def _random_email() -> str:
    """Generate a random email."""
    name = random.choice(_TR_FIRST_NAMES).lower()
    surname = random.choice(_TR_LAST_NAMES).lower()
    return f"{name}.{surname}@{random.choice(_EMAIL_DOMAINS)}"


def _random_name(pii_type: PiiType) -> str:
    """Generate a random Turkish name."""
    if pii_type == PiiType.NAME:
        return f"{random.choice(_TR_FIRST_NAMES)} {random.choice(_TR_LAST_NAMES)}"
    return f"{random.choice(_TR_FIRST_NAMES)}"


def _random_dob() -> str:
    """Generate a random date of birth."""
    y = random.randint(1950, 2010)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{d:02d}.{m:02d}.{y}"


def _random_address() -> str:
    """Generate a random Turkish address."""
    return (
        f"{random.choice(_TR_STREETS)} No:{random.randint(1, 200)}, "
        f"{random.choice(_TR_CITIES)} / Turkey"
    )


def _random_card_pan() -> str:
    """Generate a random valid Luhn PAN."""
    prefix = random.choice(["4", "5"])
    digits = [int(c) for c in prefix] + [random.randint(0, 9) for _ in range(14)]
    d = list(digits)
    for i in range(len(d) - 1, -1, -2):
        d[i] *= 2
        if d[i] > 9:
            d[i] -= 9
    check = (10 - (sum(d) % 10)) % 10
    digits.append(check)
    return "".join(str(d) for d in digits)


_PSEUDO_GENERATORS: dict[PiiType, Any] = {
    PiiType.TCKN: _random_tckn,
    PiiType.VKN: _random_vkn,
    PiiType.IBAN: _random_iban,
    PiiType.PHONE: _random_phone,
    PiiType.EMAIL: _random_email,
    PiiType.NAME: _random_name,
    PiiType.DOB: _random_dob,
    PiiType.ADDRESS: _random_address,
    PiiType.CARD_PAN: _random_card_pan,
}


# ── Hash helper ───────────────────────────────────────────────────────────────

def _compute_hash(value: str, salt: str | None = None) -> str:
    """SHA-256 hex digest of (salt + value), truncated to 12 chars."""
    raw = (salt or "") + value
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _compute_token(value: str, salt: str | None = None) -> str:
    """Deterministic token from hash, mapped to a short ID.

    NOTE: True reversible tokenization requires a vault (mapping table).
    For MVP, this is a deterministic hash prefix that can be used as an
    opaque identifier.  The same (salt + value) always produces the same token.
    """
    raw = (salt or "") + value
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    # Map first 8 hex chars to a PREFIX-NNN pattern using modulo
    num = int(h[:8], 16) % 100000
    return f"TOK-{num:05d}"


# ── Main masking functions ────────────────────────────────────────────────────

def mask_value(
    value,
    pii_type: PiiType,
    mode: MaskMode,
    show_last: int = 4,
    salt: str | None = None,
) -> str | None:
    """Mask a single PII value according to *mode*.

    Args:
        value: The raw value (scalar, will be str'd).
        pii_type: Type of PII (affects pseudonymize generation).
        mode: Masking mode.
        show_last: Number of trailing chars to show in PARTIAL mode.
        salt: Optional salt for HASH/TOKENIZE modes.

    Returns:
        Masked string, or None if input was null/NaN.
    """
    if value is None:
        return None
    if isinstance(value, float) and not value == value:  # NaN
        return None
    s = str(value).strip()
    if s == "":
        return ""

    if mode == MaskMode.FULL:
        return "[REDACTED]"

    elif mode == MaskMode.PARTIAL:
        if len(s) <= show_last:
            return s  # Too short to mask partially
        mask_len = len(s) - show_last
        return "*" * mask_len + s[mask_len:]

    elif mode == MaskMode.HASH:
        return _compute_hash(s, salt)

    elif mode == MaskMode.TOKENIZE:
        return _compute_token(s, salt)

    elif mode == MaskMode.PSEUDONYMIZE:
        gen = _PSEUDO_GENERATORS.get(pii_type)
        if gen:
            result = gen()
            # For NAME type, pass pii_type for better random name gen
            return result
        return "[PSEUDO]"

    return s  # Unknown mode, return as-is


def mask_dataframe(
    df: pd.DataFrame,
    findings: list[PiiFinding],
    mode: MaskMode,
    show_last: int = 4,
    salt: str | None = None,
) -> pd.DataFrame:
    """Return a masked COPY of *df* for the detected PII columns.

    Non-PII columns are left untouched.  The input DataFrame is never mutated.

    Args:
        df: Input DataFrame.
        findings: List of PiiFinding objects (from detect_pii).
        mode: Masking mode.
        show_last: Trailing chars to show in PARTIAL mode.
        salt: Optional salt for HASH/TOKENIZE modes.

    Returns:
        pd.DataFrame: A new, masked DataFrame (copy).
    """
    masked = df.copy()

    pii_columns = {f.column: f.pii_type for f in findings}

    for col, pii_type in pii_columns.items():
        if col not in masked.columns:
            continue
        masked[col] = masked[col].apply(
            lambda v, pt=pii_type: mask_value(v, pt, mode, show_last, salt)
        )

    return masked
