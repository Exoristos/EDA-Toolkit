"""
Veri kalitesi skorlamasi modulu.

``data_quality_score`` bir DataFrame'in genel kalitesini 0-100 arasi bir skorla
degerlendirir. Skor ne kadar yuksekse veri o kadar temizdir.

Formul (acikca denetlenebilir):
    score = 100 * (1 - 0.4*missing_rate - 0.2*duplicate_rate - 0.2*outlier_rate - 0.2*const_fraction)
    Sonuc [0, 100] araligina sikistirilir.

Agirliklandirma gerekcesi:
    - missing_rate (agirlik: 0.4): Eksik veri en buyuk kalite sorunudur.
    - duplicate_rate (agirlik: 0.2): Tekrarli satirlar analitik yanliliga yol acar.
    - outlier_rate (agirlik: 0.2): Asiri degerler istatistiksel hatalara neden olur.
    - const_fraction (agirlik: 0.2): Sabit sutunlar bilgi tasimaz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def data_quality_score(df: pd.DataFrame) -> dict:
    """DataFrame'in veri kalitesi skorunu hesapla.

    Args:
        df: Degerlendirilecek pandas DataFrame.

    Returns:
        Sozluk (dict):
            - score (float): 0-100 arasi kalite puani.
            - missing_rate (float): Hucre bazinda ortalama NaN orani.
            - duplicate_rate (float): Tekrarli satir orani.
            - outlier_rate (float): Sayisal sutunlarda aykiri deger orani
              (1.5*IQR disi).
            - constant_columns (int): Sabit (sifir varyansli) sutun sayisi.
            - components (dict): Her bir bilesenin skora etkisi.

    Ornek:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> data_quality_score(df)["score"]
        100.0
    """
    total_cols = max(len(df.columns), 1)

    # ── Eksik veri orani (missing_rate) ─────────────────────────────────────
    missing_rate = float(df.isna().mean().mean()) if not df.empty else 1.0

    # ── Tekrarli satir orani (duplicate_rate) ───────────────────────────────
    duplicate_rate = float(df.duplicated().mean()) if not df.empty else 0.0

    # ── Aykiri deger orani (outlier_rate) ───────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0 and not df.empty:
        outlier_fractions: list[float] = []
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) < 2:
                outlier_fractions.append(0.0)
                continue
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                # Butun degerler ayni; aykiri deger yok
                outlier_fractions.append(0.0)
            else:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                frac = ((col_data < lower) | (col_data > upper)).mean()
                outlier_fractions.append(float(frac))
        outlier_rate = float(np.mean(outlier_fractions))
    else:
        outlier_rate = 0.0

    # ── Sabit sutun sayisi (constant_columns) ───────────────────────────────
    if df.empty:
        constant_columns = 0
    else:
        constant_columns = sum(
            1 for col in df.columns if df[col].nunique(dropna=False) <= 1
        )
    const_fraction = constant_columns / total_cols

    # ── Agirlikli skor hesaplama ────────────────────────────────────────────
    #  missing: 0.4, duplicate: 0.2, outlier: 0.2, constant: 0.2
    raw = 1.0 - (
        0.4 * missing_rate
        + 0.2 * duplicate_rate
        + 0.2 * outlier_rate
        + 0.2 * const_fraction
    )
    score = float(max(0.0, min(100.0, round(raw * 100, 2))))

    return {
        "score": score,
        "missing_rate": round(missing_rate, 4),
        "duplicate_rate": round(duplicate_rate, 4),
        "outlier_rate": round(outlier_rate, 4),
        "constant_columns": constant_columns,
        "components": {
            "missing_penalty": round(0.4 * missing_rate * 100, 4),
            "duplicate_penalty": round(0.2 * duplicate_rate * 100, 4),
            "outlier_penalty": round(0.2 * outlier_rate * 100, 4),
            "constant_penalty": round(0.2 * const_fraction * 100, 4),
        },
    }
