"""
PII yeniden tanimlanma riski skorlamasi modulu.

``pii_risk_score`` bir veri setindeki PII bulgularina dayanarak 0-100 arasi
bir yeniden tanimlanma (re-identification) risk skoru uretir. Skor ne kadar
yuksekse risk o kadar buyuktur.

Formul (acikca denetlenebilir, NIST/ENISA/KVKK sayisal rubrik vermez):
    risk = severity_score + cardinality_score + qi_combinability + masking_contribution

Bilesenler:
    1. severity_score (0-40): PII turlerine gore en yuksek ciddiyet puani.
       TCKN=40, VKN=40, CARD_PAN=35, IBAN=30, PHONE=20, EMAIL=15, NAME=10,
       DOB=10, ADDRESS=10.
    2. cardinality_score (0-25): PII sutunlarindaki en yuksek benzersizlik
       orani (nunique/len) uzerinden hesaplanir. Yuksek kardinalite → yuksek risk.
    3. qi_combinability (0-20): Tespit edilen quasi-identifier sutunlarinin
       bir araya gelmesiyle olusan en kucuk grup boyutuna gore. Kucuk grup → yuksek risk.
    4. masking_contribution (0-15): Maskeleme seviyesine bagli risk katkisi.
       - mask_mode=None → 15 (tam acik)
       - mask_mode="full"/"hash"/"tokenize"/"pseudonymize" → 0
       - mask_mode="partial" → 8

    Toplam risk [0, 100] araligina sikistirilir.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from eda_toolkit.types import PiiFinding, PiiType

# ── PII turu ciddiyet haritasi (max 40) ───────────────────────────────────────
_PII_SEVERITY: dict[PiiType, int] = {
    PiiType.TCKN: 40,
    PiiType.VKN: 40,
    PiiType.CARD_PAN: 35,
    PiiType.IBAN: 30,
    PiiType.PHONE: 20,
    PiiType.EMAIL: 15,
    PiiType.NAME: 10,
    PiiType.DOB: 10,
    PiiType.ADDRESS: 10,
}

# ── Maskeleme risk katkisi ────────────────────────────────────────────────────
_MASKING_CONTRIBUTION: dict[str | None, int] = {
    None: 15,
    "partial": 8,
    "full": 0,
    "hash": 0,
    "tokenize": 0,
    "pseudonymize": 0,
}

# ── Quasi-identifier tespiti icin esik degerler ──────────────────────────────
_QI_MAX_UNIQUE_RATIO = 0.5  # Bir sutunun QI sayilmasi icin max unique/len
_QI_MIN_UNIQUE = 2          # En az 2 farkli deger olmali
_QI_MAX_UNIQUE_ABS = 100    # Mutlak ust sinir (cok buyuk veri setlerinde)


def _detect_qi_columns(
    df: pd.DataFrame,
    pii_columns: set[str],
) -> list[str]:
    """PII sutunlari haricindeki quasi-identifier sutunlarini tespit et.

    Dusuk kardinaliteli (kategorik/az sayida essiz deger) sutunlar QI olarak
    kabul edilir. Savunmaci yaklasim: cok dusuk kardinalite esigi kullanilir.
    """
    if df.empty:
        return []

    n = len(df)
    qi_cols: list[str] = []
    for col in df.columns:
        if col in pii_columns:
            continue
        nunique = df[col].nunique()
        if nunique < _QI_MIN_UNIQUE:
            continue
        ratio = nunique / n
        if ratio <= _QI_MAX_UNIQUE_RATIO and nunique <= _QI_MAX_UNIQUE_ABS:
            qi_cols.append(col)
    return qi_cols


def k_anonymity(df: pd.DataFrame, quasi_identifiers: list[str]) -> int:
    """k-anonimlik degerini hesapla: QI gruplarindaki en kucuk satir sayisi.

    Args:
        df: Degerlendirilecek DataFrame.
        quasi_identifiers: Quasi-identifier sutun adlari listesi.

    Returns:
        int: En kucuk grup boyutu (k degeri).

    Raises:
        ValueError: quasi_identifiers bossa veya sutunlar df'de yoksa.
    """
    if not quasi_identifiers:
        raise ValueError("Yari-tanimlayici belirtilmedi (quasi_identifiers bos)")

    missing = [c for c in quasi_identifiers if c not in df.columns]
    if missing:
        raise ValueError(
            f"Yari-tanimlayici sutun(lar) DataFrame'de bulunamadi: {missing}"
        )

    if df.empty:
        return 0

    group_sizes = df.groupby(quasi_identifiers, dropna=False).size()
    return int(group_sizes.min())


def l_diversity(
    df: pd.DataFrame,
    quasi_identifiers: list[str],
    sensitive: str,
) -> int:
    """l-cesitlilik degerini hesapla: her QI grubunda hassas sutundaki
    en az essiz deger sayisi.

    Args:
        df: Degerlendirilecek DataFrame.
        quasi_identifiers: Quasi-identifier sutun adlari listesi.
        sensitive: Hassas (duyarli) sutun adi.

    Returns:
        int: En kucuk essiz deger sayisi (l degeri).

    Raises:
        ValueError: quasi_identifiers bossa, sutunlar df'de yoksa
                    veya sensitive sutunu df'de yoksa.
    """
    if not quasi_identifiers:
        raise ValueError("Yari-tanimlayici belirtilmedi (quasi_identifiers bos)")

    missing_qi = [c for c in quasi_identifiers if c not in df.columns]
    if missing_qi:
        raise ValueError(
            f"Yari-tanimlayici sutun(lar) DataFrame'de bulunamadi: {missing_qi}"
        )

    if sensitive not in df.columns:
        raise ValueError(
            f"Hassas sutun DataFrame'de bulunamadi: '{sensitive}'"
        )

    if df.empty:
        return 0

    diversity = df.groupby(quasi_identifiers, dropna=False)[sensitive].nunique()
    return int(diversity.min())


def _compute_cardinality_score(
    df: pd.DataFrame,
    pii_columns: set[str],
) -> tuple[float, dict]:
    """PII sutunlari icindeki en yuksek kardinalite (benzersizlik) skoru.

    Returns:
        (score, details): 0-25 arasi skor ve detay sozlugu.
    """
    if not pii_columns or df.empty:
        return 0.0, {"max_uniqueness": 0.0, "max_uniqueness_column": None}

    n = len(df)
    max_uniqueness = 0.0
    max_col = None
    for col in pii_columns:
        if col not in df.columns:
            continue
        uniqueness = df[col].nunique() / n
        if uniqueness > max_uniqueness:
            max_uniqueness = uniqueness
            max_col = col

    score = min(25.0, max_uniqueness * 25.0)
    return round(score, 2), {
        "max_uniqueness": round(max_uniqueness, 4),
        "max_uniqueness_column": max_col,
    }


def _compute_qi_combinability(
    df: pd.DataFrame,
    pii_columns: set[str],
) -> tuple[float, dict]:
    """Quasi-identifier sutunlarinin birlestirilebilirlik risk skoru.

    Tespit edilen QI sutunlari uzerinde gruplama yapilir ve en kucuk grup
    boyutuna gore skor hesaplanir. Kucuk grup boyutu → yuksek risk.

    Returns:
        (score, details): 0-20 arasi skor ve detay sozlugu.
    """
    qi_cols = _detect_qi_columns(df, pii_columns)

    if len(qi_cols) < 2 or df.empty:
        return 0.0, {"qi_columns": qi_cols, "min_group_size": None}

    group_sizes = df.groupby(qi_cols, dropna=False).size()
    min_group = int(group_sizes.min())

    # Dogrusal skorlama: min_group_size >= 20 → 0, min_group_size = 1 → 19
    score = max(0.0, min(20.0, 20.0 * (1.0 - min_group / 20.0)))

    return round(score, 2), {
        "qi_columns": qi_cols,
        "min_group_size": min_group,
    }


def pii_risk_score(
    df: pd.DataFrame,
    findings: list[PiiFinding],
    mask_mode: str | None = None,
) -> dict:
    """PII bulgularina dayanarak yeniden tanimlanma risk skoru hesapla.

    Args:
        df: Degerlendirilecek DataFrame.
        findings: ``detect_pii()`` ciktisi olan PiiFinding listesi.
        mask_mode: Maskeleme modu (None, "partial", "full", "hash",
                   "tokenize", "pseudonymize").

    Returns:
        Sozluk (dict):
            - score (float): 0-100 arasi risk puani (yuksek = daha riskli).
            - factors (dict): Bilesen skorlari ve detaylari.
            - k_anonymity (int|None): PII sutunlarindan elde edilen k degeri
              (birden fazla dogrudan tanimlayici varsa).
            - max_severity_type (str|None): En yuksek ciddiyetli PII turu.
            - mask_mode (str|None): Kullanilan maskeleme modu.

    Bos bulgu listesi durumunda skor 0 doner ve not icerir.

    Ornek:
        >>> from eda_toolkit.types import PiiFinding, PiiType
        >>> import pandas as pd
        >>> df = pd.DataFrame({"tc": ["12345678901"], "ad": ["Ali"]})
        >>> findings = [
        ...     PiiFinding("tc", PiiType.TCKN, 0.95, 0.9, 1, "name+value"),
        ... ]
        >>> result = pii_risk_score(df, findings)
        >>> result["score"] > 0
        True
    """
    # ── Bos bulgu: risk yok ─────────────────────────────────────────────────
    if not findings:
        return {
            "score": 0.0,
            "factors": {
                "severity": {"score": 0.0, "max_type": None},
                "cardinality": {"score": 0.0, "max_uniqueness": 0.0, "max_uniqueness_column": None},
                "qi_combinability": {"score": 0.0, "qi_columns": [], "min_group_size": None},
                "masking_contribution": _MASKING_CONTRIBUTION.get(mask_mode, 15),
            },
            "k_anonymity": None,
            "max_severity_type": None,
            "mask_mode": mask_mode,
            "note": "PII bulgusu yok; risk skoru 0.",
        }

    # ── PII sutun kumesi ────────────────────────────────────────────────────
    pii_columns: set[str] = {f.column for f in findings if f.column in df.columns}

    # ── 1. PII turu ciddiyet skoru (0-40) ──────────────────────────────────
    max_type: PiiType | None = None
    max_severity = 0.0
    for f in findings:
        sev = _PII_SEVERITY.get(f.pii_type, 0)
        if sev > max_severity:
            max_severity = sev
            max_type = f.pii_type
    severity_score = float(max_severity)

    # ── 2. Kardinalite skoru (0-25) ────────────────────────────────────────
    cardinality_score, card_details = _compute_cardinality_score(df, pii_columns)

    # ── 3. QI birlestirilebilirlik skoru (0-20) ────────────────────────────
    qi_score, qi_details = _compute_qi_combinability(df, pii_columns)

    # ── 4. Maskeleme katkisi (0-15) ────────────────────────────────────────
    masking_contribution = _MASKING_CONTRIBUTION.get(mask_mode, 15)

    # ── Toplam skor ────────────────────────────────────────────────────────
    total = severity_score + cardinality_score + qi_score + masking_contribution
    score = round(max(0.0, min(100.0, total)), 2)

    # ── k-anonymity: birden fazla dogrudan tanimlayici varsa ───────────────
    direct_ids = [
        c for c in pii_columns
        if c in df.columns and df[c].nunique() > 1
    ]
    k_val: int | None = None
    if len(direct_ids) >= 2:
        try:
            k_val = int(df.groupby(direct_ids, dropna=False).size().min())
        except Exception:
            k_val = None

    max_type_str: str | None = max_type.value if max_type else None

    return {
        "score": score,
        "factors": {
            "severity": {
                "score": severity_score,
                "max_type": max_type_str,
            },
            "cardinality": {
                "score": cardinality_score,
                "max_uniqueness": card_details["max_uniqueness"],
                "max_uniqueness_column": card_details["max_uniqueness_column"],
            },
            "qi_combinability": {
                "score": qi_score,
                "qi_columns": qi_details["qi_columns"],
                "min_group_size": qi_details["min_group_size"],
            },
            "masking_contribution": masking_contribution,
        },
        "k_anonymity": k_val,
        "max_severity_type": max_type_str,
        "mask_mode": mask_mode,
    }
