"""
Turkish PII validators — pure functions returning (bool).

Algorithms per deepwork/interview spec and generate.py fixtures:

  TCKN (11-digit):
    ^[1-9][0-9]{10}$
    A = sum of odd-indexed digits (positions 1,3,5,7,9)
    B = sum of even-indexed digits (positions 2,4,6,8)
    d10 = (A * 7 - B) % 10
    d11 = (sum(digits[0:9]) + d10) % 10

  VKN (10-digit):
    ^\\d{10}$
    GİB algorithm (authoritative, matches python-stdnum stdnum.tr.vkn):
      s = 0
      for i, n in enumerate(reversed(first_9_digits), start=1):  # i = 1..9
          c1 = (int(n) + i) % 10
          if c1:
              c2 = (c1 * (2 ** i)) % 9 or 9
              s += c2
      check = (10 - s) % 10
    The 10th digit must equal `check`. (The "1000036101" example from earlier
    research is NOT valid by this algorithm; valid examples: 1111111114, 4540536920.)

  IBAN (26 chars TR):
    ^TR[0-9]{24}$
    Move first 4 chars to end, A=10..Z=35, MOD-97 == 1

  Phone:
    ^(+90|0)[0-9]{10}$ after stripping spaces/dashes/parens

  Email:
    Standard email regex

  Card PAN:
    13-19 digits, Luhn checksum

All functions return False on malformed input (no exceptions).
"""

from __future__ import annotations

import re

# ── Regex pre-filters (cheap before checksum) ─────────────────────────────────

_RE_TCKN = re.compile(r"^[1-9][0-9]{10}$")
_RE_VKN = re.compile(r"^[0-9]{10}$")
_RE_IBAN = re.compile(r"^TR[0-9]{24}$", re.IGNORECASE)
_RE_PHONE_RAW = re.compile(r"^(\+90|0)[0-9]{10}$")
_RE_EMAIL = re.compile(
    r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
    r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
)


def _strip_non_digits(value: str) -> str:
    """Remove all non-digit characters from a string."""
    return re.sub(r"\D", "", value)


def _strip_dialable(value: str) -> str:
    """Strip spaces, dashes, parens from a phone-like string."""
    return re.sub(r"[\s\-\(\)]", "", value)


# ── TCKN ──────────────────────────────────────────────────────────────────────

def is_valid_tckn(value) -> bool:
    """Return True if *value* is a structurally valid 11-digit TCKN.

    Validates:
      - 11 digits (after stripping non-digits)
      - First digit != 0
      - TCKN checksum (10th and 11th digits)
    """
    if value is None or (isinstance(value, float) and not value == value):  # NaN
        return False
    s = str(value).strip()
    digits_only = _strip_non_digits(s)
    if not _RE_TCKN.match(digits_only):
        return False
    digits = [int(c) for c in digits_only[:9]]
    # A = sum of odd-indexed (0,2,4,6,8 in 0-based)
    A = sum(digits[i] for i in range(0, 9, 2))
    # B = sum of even-indexed (1,3,5,7)
    B = sum(digits[i] for i in range(1, 8, 2))
    d10 = (A * 7 - B) % 10
    d11 = (sum(digits) + d10) % 10
    return d10 == int(digits_only[9]) and d11 == int(digits_only[10])


# ── VKN ───────────────────────────────────────────────────────────────────────

def is_valid_vkn(value) -> bool:
    """Return True if *value* is a structurally valid 10-digit VKN.

    Uses the authoritative GİB checksum algorithm (matches python-stdnum
    ``stdnum.tr.vkn``): iterate the first 9 digits in reverse with i=1..9,
    c1 = (digit + i) % 10; if c1 != 0 add (c1 * 2**i) % 9 (or 9); the 10th
    digit must equal (10 - sum) % 10.
    """
    if value is None or (isinstance(value, float) and not value == value):
        return False
    digits_only = _strip_non_digits(str(value).strip())
    if not _RE_VKN.match(digits_only):
        return False
    s = 0
    for i, n in enumerate(reversed(digits_only[:9]), 1):
        c1 = (int(n) + i) % 10
        if c1:
            c2 = (c1 * (2 ** i)) % 9 or 9
            s += c2
    expected = (10 - s) % 10
    return expected == int(digits_only[9])


# ── IBAN ──────────────────────────────────────────────────────────────────────

def _iban_to_numeric(iban: str) -> int:
    """Convert IBAN string to integer for MOD-97."""
    rearranged = iban[4:] + iban[:4]
    numeric_parts: list[str] = []
    for ch in rearranged:
        if ch.isalpha():
            numeric_parts.append(str(ord(ch) - 55))
        else:
            numeric_parts.append(ch)
    return int("".join(numeric_parts))


def is_valid_iban(value) -> bool:
    """Return True if *value* is a valid 26-character TR IBAN (MOD-97 == 1).

    Strips spaces and uppercases before validation.
    """
    if value is None or (isinstance(value, float) and not value == value):
        return False
    s = str(value).strip()
    cleaned = s.replace(" ", "").upper()
    if not _RE_IBAN.match(cleaned):
        return False
    try:
        return _iban_to_numeric(cleaned) % 97 == 1
    except ValueError:
        return False


# ── Phone ─────────────────────────────────────────────────────────────────────

def is_valid_phone(value) -> bool:
    """Return True if *value* matches Turkish phone format (+90|0) + 10 digits."""
    if value is None or (isinstance(value, float) and not value == value):
        return False
    s = str(value).strip()
    cleaned = _strip_dialable(s)
    return bool(_RE_PHONE_RAW.match(cleaned))


# ── Email ─────────────────────────────────────────────────────────────────────

def is_valid_email(value) -> bool:
    """Return True if *value* matches standard email format."""
    if value is None:
        return False
    s = str(value).strip()
    if len(s) > 254:
        return False
    return bool(_RE_EMAIL.match(s))


# ── Card PAN (Luhn) ──────────────────────────────────────────────────────────

def _luhn_checksum_digit(digits: list[int]) -> int:
    """Luhn checksum digit for the given partial PAN digits (excluding check)."""
    d = list(digits)
    for i in range(len(d) - 1, -1, -2):
        d[i] *= 2
        if d[i] > 9:
            d[i] -= 9
    return (10 - (sum(d) % 10)) % 10


def is_valid_card_pan(value) -> bool:
    """Return True if *value* is a 13-19 digit Luhn-valid card PAN.

    Strips spaces and dashes before validation.
    """
    if value is None or (isinstance(value, float) and not value == value):
        return False
    s = str(value).strip()
    cleaned = re.sub(r"[\s\-]", "", s)
    if not cleaned.isdigit() or not (13 <= len(cleaned) <= 19):
        return False
    digits = [int(c) for c in cleaned[:-1]]
    expected = _luhn_checksum_digit(digits)
    return expected == int(cleaned[-1])


# ── Generic helper ────────────────────────────────────────────────────────────

def match_rate(values, validator) -> float:
    """Fraction of non-null *values* for which ``validator(v)`` returns True.

    Args:
        values: iterable of scalar values (may contain None/NaN/empty).
        validator: callable ``(value) -> bool``.

    Returns:
        float in [0, 1]; returns 0.0 if no non-null values.
    """
    non_null = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and not v == v:  # NaN
            continue
        s = str(v).strip()
        if s == "":
            continue
        non_null.append(s)
    if not non_null:
        return 0.0
    passed = sum(1 for v in non_null if validator(v))
    return passed / len(non_null)
