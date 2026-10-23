"""Phase 4 — Turkish PII detection and masking tests.

Uses existing fixtures:
  - tests/fixtures/vectors/*.json  (valid+invalid TCKN/VKN/IBAN/PAN)
  - tests/fixtures/data/mixed_with_pii.csv
  - tests/fixtures/data/numeric_heavy.csv
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from eda_toolkit.pii.validators import (
    is_valid_tckn,
    is_valid_vkn,
    is_valid_iban,
    is_valid_phone,
    is_valid_email,
    is_valid_card_pan,
    match_rate,
)
from eda_toolkit.pii.detector import detect_pii
from eda_toolkit.pii.masker import mask_value, mask_dataframe
from eda_toolkit.types import MaskMode, PiiFinding, PiiType

# ── Helpers ───────────────────────────────────────────────────────────────────

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
VECTORS_DIR = os.path.join(FIXTURES_DIR, "vectors")
DATA_DIR = os.path.join(FIXTURES_DIR, "data")


def _load_vectors(filename: str) -> tuple[list[str], list[str]]:
    path = os.path.join(VECTORS_DIR, filename)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["valid"], data["invalid"]


# ── TCKN validator ────────────────────────────────────────────────────────────

TCKN_VALID, TCKN_INVALID = _load_vectors("tckn_vectors.json")


@pytest.mark.parametrize("value", TCKN_VALID)
def test_is_valid_tckn_valid(value):
    assert is_valid_tckn(value), f"expected valid TCKN: {value}"


@pytest.mark.parametrize("value", TCKN_INVALID)
def test_is_valid_tckn_invalid(value):
    assert not is_valid_tckn(value), f"expected invalid TCKN: {value}"


def test_is_valid_tckn_edge_cases():
    """Edge cases: NaN, None, empty, non-string, formatted."""
    import math
    assert not is_valid_tckn(None)
    assert not is_valid_tckn(float("nan"))
    assert not is_valid_tckn("")
    assert not is_valid_tckn(123)  # too short
    assert is_valid_tckn(10000000146)  # valid as int
    assert is_valid_tckn("10000000146")


# ── VKN validator ─────────────────────────────────────────────────────────────

VKN_VALID, VKN_INVALID = _load_vectors("vkn_vectors.json")


@pytest.mark.parametrize("value", VKN_VALID)
def test_is_valid_vkn_valid(value):
    assert is_valid_vkn(value), f"expected valid VKN: {value}"


@pytest.mark.parametrize("value", VKN_INVALID)
def test_is_valid_vkn_invalid(value):
    assert not is_valid_vkn(value), f"expected invalid VKN: {value}"


def test_is_valid_vkn_edge_cases():
    assert not is_valid_vkn(None)
    assert not is_valid_vkn(float("nan"))
    assert not is_valid_vkn("")
    assert not is_valid_vkn("123456789")   # 9 digits
    assert not is_valid_vkn("12345678901") # 11 digits
    assert not is_valid_vkn("ABC0000000")


# ── IBAN validator ────────────────────────────────────────────────────────────

IBAN_VALID, IBAN_INVALID = _load_vectors("iban_vectors.json")


@pytest.mark.parametrize("value", IBAN_VALID)
def test_is_valid_iban_valid(value):
    assert is_valid_iban(value), f"expected valid IBAN: {value}"


@pytest.mark.parametrize("value", IBAN_INVALID)
def test_is_valid_iban_invalid(value):
    assert not is_valid_iban(value), f"expected invalid IBAN: {value}"


def test_is_valid_iban_spaces():
    """IBAN with spaces should pass."""
    assert is_valid_iban("TR47 0000 1001 0000 0350 9300 01")
    assert is_valid_iban("tr470000100100000350930001")  # lowercase


# ── Phone validator ───────────────────────────────────────────────────────────

_PHONE_VALID = [
    "+905321112233",
    "05321112233",
    "+90 532 111 22 33",
    "0(532)1112233",
    "+90-532-111-22-33",
]
_PHONE_INVALID = [
    "12345",           # too short
    "+105321112233",   # wrong country code
    "5",               # too short
    "0532111223",      # 10 digits after 0 but total is 11 (0 + 9 digits)
    "+9053211122",     # too short
    "abc",
    "",
    None,
]


@pytest.mark.parametrize("value", _PHONE_VALID)
def test_is_valid_phone_valid(value):
    assert is_valid_phone(value), f"expected valid phone: {value}"


@pytest.mark.parametrize("value", _PHONE_INVALID)
def test_is_valid_phone_invalid(value):
    assert not is_valid_phone(value), f"expected invalid phone: {value}"


# ── Email validator ───────────────────────────────────────────────────────────

_EMAIL_VALID = [
    "test@example.com",
    "user.name@domain.co.uk",
    "a@b.com",
    "first.last@sub.domain.org",
]
_EMAIL_INVALID = [
    "notanemail",
    "@no-local.com",
    "no-at.com",
    "spaces in@domain.com",
    "",
    "x" * 300,  # too long for email (RFC 5321 limit is 254)
]


@pytest.mark.parametrize("value", _EMAIL_VALID)
def test_is_valid_email_valid(value):
    assert is_valid_email(value), f"expected valid email: {value}"


@pytest.mark.parametrize("value", _EMAIL_INVALID)
def test_is_valid_email_invalid(value):
    assert not is_valid_email(value), f"expected invalid email: {value}"


# ── Card PAN validator ────────────────────────────────────────────────────────

PAN_VALID, PAN_INVALID = _load_vectors("pan_vectors.json")

@pytest.mark.parametrize("value", PAN_VALID)
def test_is_valid_card_pan_valid(value):
    assert is_valid_card_pan(value), f"expected valid PAN: {value}"


@pytest.mark.parametrize("value", PAN_INVALID)
def test_is_valid_card_pan_invalid(value):
    # Empty string is special: isdigit() returns False for ""
    if value == "":
        assert not is_valid_card_pan(value)
    else:
        assert not is_valid_card_pan(value), f"expected invalid PAN: {value}"


def test_is_valid_card_pan_formatted():
    """PAN with spaces/dashes should pass."""
    assert is_valid_card_pan("5012 3020 0266 9609")
    assert is_valid_card_pan("5012-3020-0266-9609")


# ── match_rate helper ─────────────────────────────────────────────────────────

def test_match_rate_all_pass():
    values = ["10000000146", "70453496200", "61648298228"]
    assert match_rate(values, is_valid_tckn) == 1.0


def test_match_rate_none_pass():
    values = ["00000000000", "12345678901", "99999999999"]
    assert match_rate(values, is_valid_tckn) == 0.0


def test_match_rate_mixed():
    values = ["10000000146", "bad", None, "", "70453496200"]
    # 2 valid out of 3 non-null-nonempty
    assert match_rate(values, is_valid_tckn) == 2.0 / 3.0


def test_match_rate_nan():
    import math
    values = [float("nan"), None, ""]
    assert match_rate(values, is_valid_tckn) == 0.0


# ── Detector on mixed_with_pii.csv ────────────────────────────────────────────

def test_detect_pii_mixed_csv(mixed_with_pii_csv):
    """detect_pii should find tc_kimlik (TCKN), iban (IBAN), telefon (PHONE),
    email (EMAIL), ad/soyad (NAME) with reasonable match rates."""
    df = pd.read_csv(mixed_with_pii_csv, dtype=str, keep_default_na=False)
    findings = detect_pii(df, sample_size=50, threshold=0.8)

    found_cols = {f.column: f for f in findings}

    # Assert at least these columns are detected
    assert "tc_kimlik" in found_cols, f"Expected tc_kimlik in findings, got: {list(found_cols)}"
    assert "iban" in found_cols, f"Expected iban in findings"
    assert "telefon" in found_cols, f"Expected telefon in findings"
    assert "email" in found_cols, f"Expected email in findings"

    # Name columns should be detected by column-name heuristics
    name_cols = [f for f in findings if f.pii_type == PiiType.NAME]
    assert len(name_cols) >= 1, "Expected at least 1 NAME finding"

    # Verify types
    assert found_cols["tc_kimlik"].pii_type == PiiType.TCKN
    assert found_cols["iban"].pii_type == PiiType.IBAN
    assert found_cols["telefon"].pii_type == PiiType.PHONE
    assert found_cols["email"].pii_type == PiiType.EMAIL

    # Verify reasonable match rates
    assert found_cols["tc_kimlik"].match_rate >= 0.8
    assert found_cols["iban"].match_rate >= 0.8
    assert found_cols["email"].match_rate >= 0.8

    # Detection layers should include value-layer hits for checksum types
    assert "value" in found_cols["tc_kimlik"].detection_layer


# ── Detector on numeric_heavy.csv ─────────────────────────────────────────────

def test_detect_pii_numeric_heavy_csv(numeric_heavy_csv):
    """detect_pii on numeric_heavy should find NO PII (empty or near-empty)."""
    df = pd.read_csv(numeric_heavy_csv, dtype=str, keep_default_na=False)
    findings = detect_pii(df, sample_size=100, threshold=0.8)

    # Should be empty or have only very low-confidence entropy-based false hits
    # that shouldn't pass the 0.8 threshold
    high_conf = [f for f in findings if f.confidence >= 0.5]
    assert len(high_conf) == 0, (
        f"Expected 0 high-confidence PII findings in numeric_heavy, "
        f"got: {[(f.column, f.pii_type.value, f.confidence) for f in high_conf]}"
    )


# ── Masker: mask_value modes ──────────────────────────────────────────────────

def test_mask_partial():
    """Partial masking shows last N chars."""
    assert mask_value("1234567890", PiiType.TCKN, MaskMode.PARTIAL, show_last=4) == "******7890"
    assert mask_value("ab", PiiType.NAME, MaskMode.PARTIAL, show_last=4) == "ab"  # too short
    assert mask_value("abcdef", PiiType.TCKN, MaskMode.PARTIAL, show_last=0) == "******"


def test_mask_full():
    assert mask_value("1234567890", PiiType.TCKN, MaskMode.FULL) == "[REDACTED]"
    assert mask_value("test@example.com", PiiType.EMAIL, MaskMode.FULL) == "[REDACTED]"


def test_mask_hash():
    """Hash is deterministic and changes with salt."""
    h1 = mask_value("test", PiiType.EMAIL, MaskMode.HASH, salt="s1")
    h2 = mask_value("test", PiiType.EMAIL, MaskMode.HASH, salt="s1")
    h3 = mask_value("test", PiiType.EMAIL, MaskMode.HASH, salt="s2")

    assert h1 == h2  # Same input + same salt → same hash
    assert h1 != h3  # Different salt → different hash
    assert len(h1) == 12  # Truncated hex


def test_mask_tokenize():
    """Tokenize is deterministic."""
    t1 = mask_value("test", PiiType.EMAIL, MaskMode.TOKENIZE, salt="s1")
    t2 = mask_value("test", PiiType.EMAIL, MaskMode.TOKENIZE, salt="s1")
    assert t1 == t2
    assert t1.startswith("TOK-")


def test_mask_pseudonymize_tckn():
    """Pseudonymized TCKN should pass validation."""
    p = mask_value("10000000146", PiiType.TCKN, MaskMode.PSEUDONYMIZE)
    assert p is not None
    assert is_valid_tckn(p), f"Pseudonymized TCKN {p} should be valid"


def test_mask_pseudonymize_iban():
    """Pseudonymized IBAN should pass validation."""
    p = mask_value("TR470000100100000350930001", PiiType.IBAN, MaskMode.PSEUDONYMIZE)
    assert p is not None
    assert is_valid_iban(p), f"Pseudonymized IBAN {p} should be valid"


def test_mask_pseudonymize_vkn():
    """Pseudonymized VKN should pass weighted-mod validation."""
    p = mask_value("8705564261", PiiType.VKN, MaskMode.PSEUDONYMIZE)
    assert p is not None
    assert is_valid_vkn(p), f"Pseudonymized VKN {p} should be valid"


def test_mask_pseudonymize_card_pan():
    """Pseudonymized PAN should pass Luhn."""
    p = mask_value("1111222233334444", PiiType.CARD_PAN, MaskMode.PSEUDONYMIZE)
    assert p is not None
    assert is_valid_card_pan(p), f"Pseudonymized PAN {p} should be valid"


def test_mask_pseudonymize_phone():
    """Pseudonymized phone should pass phone validation."""
    for _ in range(10):
        p = mask_value("05321112233", PiiType.PHONE, MaskMode.PSEUDONYMIZE)
        assert p is not None
        assert is_valid_phone(p), f"Pseudonymized phone {p} should be valid"


def test_mask_null():
    """Null/NaN should return None."""
    import math
    assert mask_value(None, PiiType.TCKN, MaskMode.FULL) is None
    assert mask_value(float("nan"), PiiType.TCKN, MaskMode.FULL) is None
    assert mask_value("", PiiType.TCKN, MaskMode.FULL) == ""


# ── mask_dataframe ────────────────────────────────────────────────────────────

def test_mask_dataframe_returns_copy(mixed_with_pii_csv):
    """mask_dataframe returns a copy, input is untouched."""
    df = pd.read_csv(mixed_with_pii_csv, dtype=str, keep_default_na=False)
    original = df.copy()

    findings = detect_pii(df, sample_size=50, threshold=0.8)
    masked = mask_dataframe(df, findings, MaskMode.FULL)

    # Input unchanged
    assert_frame_equal(df, original)

    # masked is a different object
    assert masked is not df

    # PII columns should be masked
    tc_finding = [f for f in findings if f.pii_type == PiiType.TCKN]
    if tc_finding:
        col = tc_finding[0].column
        # Check a sample value was masked
        for v in masked[col]:
            if v and str(v).strip():
                assert str(v) == "[REDACTED]", f"Expected [REDACTED] but got {v}"


def test_mask_dataframe_non_pii_untouched(mixed_with_pii_csv):
    """Non-PII columns should remain unchanged."""
    df = pd.read_csv(mixed_with_pii_csv, dtype=str, keep_default_na=False)
    findings = detect_pii(df, sample_size=50, threshold=0.8)
    masked = mask_dataframe(df, findings, MaskMode.FULL)

    non_pii = [c for c in df.columns if c not in {f.column for f in findings}]
    for col in non_pii:
        assert df[col].equals(masked[col]), f"Column {col} should be unchanged"


# ── Pseudo-generator consistency ──────────────────────────────────────────────

def test_pseudo_tckn_always_valid():
    """Repeated pseudo-generation should always produce valid TCKNs."""
    from eda_toolkit.pii.masker import _random_tckn
    for _ in range(50):
        v = _random_tckn()
        assert is_valid_tckn(v), f"Generated invalid TCKN: {v}"


def test_pseudo_iban_always_valid():
    """Repeated pseudo-generation should always produce valid IBANs."""
    from eda_toolkit.pii.masker import _random_iban
    for _ in range(20):
        v = _random_iban()
        assert is_valid_iban(v), f"Generated invalid IBAN: {v}"


def test_pseudo_pan_always_valid():
    """Repeated pseudo-generation should always produce valid PANs."""
    from eda_toolkit.pii.masker import _random_card_pan
    for _ in range(20):
        v = _random_card_pan()
        assert is_valid_card_pan(v), f"Generated invalid PAN: {v}"


def test_pseudo_vkn_always_valid():
    """Repeated pseudo-generation should always produce valid VKNs."""
    from eda_toolkit.pii.masker import _random_vkn
    for _ in range(20):
        v = _random_vkn()
        assert is_valid_vkn(v), f"Generated invalid VKN: {v}"
