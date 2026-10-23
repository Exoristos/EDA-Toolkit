"""Tests for eda_toolkit.analytics.risk — pii_risk_score, k_anonymity, l_diversity."""

from __future__ import annotations

import pandas as pd
import pytest

from eda_toolkit.analytics.risk import (
    k_anonymity,
    l_diversity,
    pii_risk_score,
)
from eda_toolkit.types import PiiFinding, PiiType


# ── Inline fixtures (MF-1: no conftest.py edits) ──────────────────────────────

def _make_finding(
    column: str,
    pii_type: PiiType,
    confidence: float = 0.95,
    match_rate: float = 0.9,
    sample_count: int = 10,
    detection_layer: str = "name+value",
) -> PiiFinding:
    """Yardimci: PiiFinding olustur."""
    return PiiFinding(
        column=column,
        pii_type=pii_type,
        confidence=confidence,
        match_rate=match_rate,
        sample_count=sample_count,
        detection_layer=detection_layer,
    )


def _small_df_tckn() -> pd.DataFrame:
    """TCKN sutunu iceren kucuk DataFrame."""
    return pd.DataFrame({
        "tc_kimlik": ["12345678901", "23456789012", "34567890123"],
        "isim": ["Ali", "Veli", "Ayse"],
        "il": ["Ankara", "Istanbul", "Ankara"],
        "yas": [25, 30, 28],
    })


def _df_with_qi() -> pd.DataFrame:
    """Quasi-identifier sutunlari iceren DataFrame."""
    return pd.DataFrame({
        "tc_kimlik": ["11111111111", "22222222222", "33333333333",
                      "44444444444", "55555555555", "66666666666"],
        "il": ["Ankara", "Ankara", "Ankara", "Istanbul", "Istanbul", "Istanbul"],
        "cinsiyet": ["E", "E", "K", "E", "K", "K"],
        "gelir": [5000, 5200, 4800, 7000, 7200, 6800],
    })


def _df_k_anon() -> pd.DataFrame:
    """k-anonymity testi icin bilinen DataFrame."""
    return pd.DataFrame({
        "il": ["Ankara", "Ankara", "Ankara", "Istanbul", "Istanbul"],
        "cinsiyet": ["E", "E", "K", "E", "K"],
        "deger": [10, 20, 30, 40, 50],
    })


# ═══════════════════════════════════════════════════════════════════════════════
# k_anonymity tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestKAnonymity:
    """k_anonymity fonksiyonu testleri."""

    def test_known_k_value(self):
        """Bilinen veride k degeri elle hesaplananla ayni olmali."""
        df = _df_k_anon()
        # Ankara-E: 2, Ankara-K: 1, Istanbul-E: 1, Istanbul-K: 1
        # min group size = 1
        result = k_anonymity(df, ["il", "cinsiyet"])
        assert result == 1

    def test_larger_k(self):
        """k > 1 durumu."""
        df = pd.DataFrame({
            "bolge": ["X", "X", "X", "Y", "Y", "Y"],
            "tip":   ["A", "A", "A", "B", "B", "B"],
        })
        # Each group has 3 rows → k = 3
        result = k_anonymity(df, ["bolge", "tip"])
        assert result == 3

    def test_single_qi(self):
        """Tek QI sutunu ile calismali."""
        df = pd.DataFrame({
            "sehir": ["A", "A", "A", "B"],
        })
        # A: 3, B: 1 → min = 1
        result = k_anonymity(df, ["sehir"])
        assert result == 1

    def test_raises_on_empty_qi(self):
        """Bos QI listesi ValueError firlatmali."""
        df = _df_k_anon()
        with pytest.raises(ValueError, match="Yari-tanimlayici belirtilmedi"):
            k_anonymity(df, [])

    def test_raises_on_missing_column(self):
        """DataFrame'de olmayan sutun ValueError firlatmali."""
        df = _df_k_anon()
        with pytest.raises(ValueError, match="bulunamadi"):
            k_anonymity(df, ["il", "varolmayan_sutun"])

    def test_empty_df(self):
        """Bos DataFrame 0 dondurmeli."""
        df = pd.DataFrame({"x": []}, dtype="int64")
        result = k_anonymity(df, ["x"])
        assert result == 0


# ═══════════════════════════════════════════════════════════════════════════════
# l_diversity tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestLDiversity:
    """l_diversity fonksiyonu testleri."""

    def test_known_l_value(self):
        """Bilinen veride l degeri elle hesaplananla ayni olmali."""
        df = pd.DataFrame({
            "il": ["Ankara", "Ankara", "Ankara", "Istanbul", "Istanbul"],
            "cinsiyet": ["E", "E", "K", "E", "K"],
            "hastalik": ["Grip", "Grip", "Migren", "Grip", "Soguk_alginligi"],
        })
        # Ankara-E: ["Grip", "Grip"] → 1 unique
        # Ankara-K: ["Migren"] → 1
        # Istanbul-E: ["Grip"] → 1
        # Istanbul-K: ["Soguk_alginligi"] → 1
        # min = 1
        result = l_diversity(df, ["il", "cinsiyet"], "hastalik")
        assert result == 1

    def test_larger_l(self):
        """l > 1 durumu."""
        df = pd.DataFrame({
            "bolge": ["X", "X", "X", "Y", "Y", "Y"],
            "tip":   ["A", "A", "B", "A", "B", "B"],
            "sonuc": ["p", "q", "r", "p", "q", "r"],
        })
        # X-A: ["p", "q"] → 2, X-B: ["r"] → 1, Y-A: ["p"] → 1, Y-B: ["q", "r"] → 2
        # min = 1
        result = l_diversity(df, ["bolge", "tip"], "sonuc")
        assert result == 1

    def test_l_diversity_all_unique(self):
        """Tum gruplarda sensitive degerler farkliysa l > 1."""
        df = pd.DataFrame({
            "sehir": ["A", "A", "B", "B"],
            "hasta": ["X", "Y", "Z", "W"],
        })
        # A: 2 unique, B: 2 unique → l = 2
        result = l_diversity(df, ["sehir"], "hasta")
        assert result == 2

    def test_raises_on_empty_qi(self):
        """Bos QI listesi ValueError firlatmali."""
        df = pd.DataFrame({"x": [1], "y": [2]})
        with pytest.raises(ValueError, match="Yari-tanimlayici belirtilmedi"):
            l_diversity(df, [], "y")

    def test_raises_on_missing_sensitive(self):
        """Hassas sutun yoksa ValueError firlatmali."""
        df = pd.DataFrame({"x": [1], "y": [2]})
        with pytest.raises(ValueError, match="Hassas sutun"):
            l_diversity(df, ["x"], "olmayan")

    def test_raises_on_missing_qi_column(self):
        """QI sutunu yoksa ValueError firlatmali."""
        df = pd.DataFrame({"x": [1], "y": [2]})
        with pytest.raises(ValueError, match="bulunamadi"):
            l_diversity(df, ["olmayan"], "y")


# ═══════════════════════════════════════════════════════════════════════════════
# pii_risk_score tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPiiRiskScore:
    """pii_risk_score fonksiyonu testleri."""

    def test_empty_findings_score_zero(self):
        """Bos bulgu listesi → skor 0."""
        df = _small_df_tckn()
        result = pii_risk_score(df, [])
        assert result["score"] == 0.0
        assert result["note"] is not None
        assert result["max_severity_type"] is None
        assert result["k_anonymity"] is None

    def test_tckn_no_mask_high_score(self):
        """TCKN bulgusu + maskesiz → yuksek risk skoru."""
        df = _small_df_tckn()
        findings = [_make_finding("tc_kimlik", PiiType.TCKN)]
        result = pii_risk_score(df, findings, mask_mode=None)
        # severity=40 + card=~25(3 unique/3) + qi=0(sadece il, 1 QI yetersiz) + mask=15
        # = 80 civari
        assert result["score"] >= 60.0, f"Expected >=60, got {result['score']}"
        assert result["max_severity_type"] == "tckn"
        assert result["mask_mode"] is None
        assert result["factors"]["masking_contribution"] == 15

    def test_masked_score_strictly_lower(self):
        """mask_mode="full" ile skor maskesize gore KESINLIKLE dusuk olmali (monotoniklik)."""
        df = _small_df_tckn()
        findings = [_make_finding("tc_kimlik", PiiType.TCKN)]

        unmasked = pii_risk_score(df, findings, mask_mode=None)
        masked = pii_risk_score(df, findings, mask_mode="full")

        assert masked["score"] < unmasked["score"], (
            f"Masked score ({masked['score']}) must be < unmasked ({unmasked['score']})"
        )

    def test_partial_mask_between_none_and_full(self):
        """partial maskeleme skoru, maskesiz ile full arasinda olmali."""
        df = _small_df_tckn()
        findings = [_make_finding("tc_kimlik", PiiType.TCKN)]

        unmasked = pii_risk_score(df, findings, mask_mode=None)
        partial = pii_risk_score(df, findings, mask_mode="partial")
        full = pii_risk_score(df, findings, mask_mode="full")

        assert full["score"] < partial["score"] < unmasked["score"], (
            f"Expected full({full['score']}) < partial({partial['score']}) < unmasked({unmasked['score']})"
        )

    def test_return_structure(self):
        """Donus sozlugu tum anahtarlari icermeli."""
        df = _small_df_tckn()
        findings = [_make_finding("tc_kimlik", PiiType.TCKN)]
        result = pii_risk_score(df, findings)

        assert "score" in result
        assert "factors" in result
        assert "k_anonymity" in result
        assert "max_severity_type" in result
        assert "mask_mode" in result
        assert 0.0 <= result["score"] <= 100.0

        factors = result["factors"]
        assert "severity" in factors
        assert "cardinality" in factors
        assert "qi_combinability" in factors
        assert "masking_contribution" in factors

    def test_hash_tokenize_pseudonymize_same_as_full(self):
        """hash, tokenize, pseudonymize → full ile ayni maskeleme katkisi."""
        df = _small_df_tckn()
        findings = [_make_finding("tc_kimlik", PiiType.TCKN)]

        full = pii_risk_score(df, findings, mask_mode="full")
        for mode in ("hash", "tokenize", "pseudonymize"):
            result = pii_risk_score(df, findings, mask_mode=mode)
            assert result["score"] == full["score"], (
                f"{mode} score ({result['score']}) != full ({full['score']})"
            )
            assert result["factors"]["masking_contribution"] == 0

    def test_multiple_findings_max_severity(self):
        """Birden fazla PII turunde en yuksek ciddiyet esasi."""
        df = pd.DataFrame({
            "eposta": ["a@b.com", "c@d.com"],
            "tc": ["12345678901", "23456789012"],
        })
        # hem EMAIL (15) hem TCKN (40) → max = 40
        findings = [
            _make_finding("eposta", PiiType.EMAIL),
            _make_finding("tc", PiiType.TCKN, confidence=0.95),
        ]
        result = pii_risk_score(df, findings, mask_mode="full")
        assert result["max_severity_type"] == "tckn"
        assert result["factors"]["severity"]["score"] == 40.0

    def test_qi_combinability_detected(self):
        """QI sutunlari birlestirilebilirlik skoru hesaplanmali."""
        df = _df_with_qi()
        findings = [_make_finding("tc_kimlik", PiiType.TCKN)]
        result = pii_risk_score(df, findings)
        qi_info = result["factors"]["qi_combinability"]
        # il ve cinsiyet QI olarak tespit edilmeli
        assert len(qi_info["qi_columns"]) >= 1
        assert qi_info["min_group_size"] is not None

    def test_score_range_always_valid(self):
        """Skor her durumda [0, 100] araliginda."""
        df = _small_df_tckn()
        findings = [_make_finding("tc_kimlik", PiiType.TCKN)]

        for mode in [None, "partial", "full", "hash", "tokenize", "pseudonymize"]:
            result = pii_risk_score(df, findings, mask_mode=mode)
            assert 0.0 <= result["score"] <= 100.0, (
                f"mode={mode} score={result['score']} out of range"
            )

    def test_all_pii_types_coverage(self):
        """Tum PII turleri icin calismali (cokme yok)."""
        df = pd.DataFrame({
            "col": ["test"] * 5,
        })
        for pii_type in PiiType:
            findings = [_make_finding("col", pii_type)]
            result = pii_risk_score(df, findings)
            assert result["score"] >= 0.0

    def test_df_with_no_matching_pii_columns(self):
        """Bulgudaki sutun DataFrame'de yoksa sorun cikmamali."""
        df = pd.DataFrame({"x": [1, 2]})
        findings = [_make_finding("varolmayan_sutun", PiiType.TCKN)]
        result = pii_risk_score(df, findings)
        # severity hala 40, card=0 (sutun yok), mask=None=15 → 55
        assert result["score"] >= 0.0
        assert result["factors"]["cardinality"]["score"] == 0.0


def test_unknown_mask_mode_defaults_to_unmasked():
    """NH-6 (ora-9): bilinmeyen mask_mode => 'maske yok' gibi davranir (15)."""
    df = pd.DataFrame({"tc": [f"{i:011d}" for i in range(100, 200)]})
    findings = [_make_finding("tc", PiiType.TCKN)]
    unknown = pii_risk_score(df, findings, mask_mode="bilinmeyen_mod")
    unmasked = pii_risk_score(df, findings, mask_mode=None)
    assert unknown["score"] == pytest.approx(unmasked["score"])
