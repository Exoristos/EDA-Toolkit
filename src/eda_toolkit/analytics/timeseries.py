"""Zaman serisi ayrıştırma ve tahmin (statsmodels).

Her fonksiyon bir dict döndürür. Olası hatalar raise edilmez; ``{"error": ...}``
sözlüğü ile döner, böylece pipeline veya dashboard asla bir seri yüzünden çökmez.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def decompose(
    df: pd.DataFrame, col: str, period: int | None = None
) -> dict:
    """Toplamsal zaman serisi ayrıştırması (trend + mevsimsellik + kalıntı).

    Args:
        df: En az bir sayısal sütun içeren DataFrame. Satır indeksinin
            ``DatetimeIndex`` ya da ardışık tam sayı indeksi olması beklenir;
            değilse yeniden indeksleme denenir.
        col: Ayrıştırılacak sayısal sütun adı.
        period: Mevsimsellik periyodu (None ise sezgisel varsayılan denenir;
            hiçbiri yoksa hata döner).

    Returns:
        ``{"trend": [...], "seasonal": [...], "residual": [...], "period": int,
        "n": int}``  ya da  ``{"error": str}``.
        NaN değerler ``None`` ile değiştirilir.
    """
    # ── Doğrulama ──
    if col not in df.columns:
        return {"error": f"Sutun '{col}' DataFrame icinde bulunamadi."}
    series = df[col]
    if not pd.api.types.is_numeric_dtype(series):
        return {"error": f"'{col}' sutunu sayisal degil."}
    series = series.dropna()
    if len(series) < 3:
        return {"error": f"Yetersiz gozlem: {len(series)} (en az 3 gerekir)."}

    # ── İndeks düzeltme ──
    if isinstance(series.index, pd.DatetimeIndex):
        pass  # ideal
    else:
        try:
            # Basit tam sayı indekse zorla
            series = series.reset_index(drop=True)
        except Exception:
            pass

    # ── Periyot belirleme ──
    if period is None:
        period = _guess_period(series)
        if period is None:
            return {
                "error": "Periyot belirlenemedi; lutfen period parametresi verin."
            }

    # ── Yeterli uzunluk kontrolü ──
    n = len(series)
    if n < 2 * period:
        return {
            "error": (
                f"Seri cok kisa: {n} gozlem, ancak 2*period={2 * period} gerekiyor."
            ),
            "period": period,
        }

    # ── Ayrıştırma ──
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose

        result = seasonal_decompose(
            series.values, model="additive", period=period, extrapolate_trend=0
        )
    except Exception as e:
        return {"error": f"Ayrıştırma başarısız: {e}", "period": period}

    # NaN → None dönüşümü
    def _tolist(arr) -> list:
        if arr is None:
            return []
        lst = arr.tolist()
        return [None if (isinstance(v, float) and np.isnan(v)) else v for v in lst]

    return {
        "trend": _tolist(result.trend),
        "seasonal": _tolist(result.seasonal),
        "residual": _tolist(result.resid),
        "period": period,
        "n": n,
    }


def forecast(
    df: pd.DataFrame,
    col: str,
    horizon: int,
    method: str = "arima",
    period: int | None = None,
) -> dict:
    """Zaman serisi nokta tahmini ve %95 güven aralığı.

    Args:
        df: En az bir sayısal sütun içeren DataFrame.
        col: Tahmin yapılacak sayısal sütun adı.
        horizon: İleri adım sayısı.
        method: ``"arima"`` (varsayılan, mevsimsel olmayan) veya ``"sarimax"``
            (mevsimsel, ``period`` gerektirir). ``method="arima"`` ile ``period``
            verilirse otomatik olarak SARIMAX'a yükseltilir ve uyarı verilir.
        period: Mevsimsellik periyodu (sadece ``method="sarimax"`` için gereklidir;
            ``method="arima"`` ile verilirse yükseltme tetiklenir).

    Returns:
        ``{"point": [...], "ci_lower": [...], "ci_upper": [...],
        "method": str, "horizon": int, "order": str}`` ya da ``{"error": str}``.

    Raises:
        ValueError: Bilinmeyen ``method`` değeri.
    """
    # ── Metot doğrulama ──
    allowed = {"arima", "sarimax"}
    if method not in allowed:
        raise ValueError(
            f"Bilinmeyen tahmin yontemi: '{method}'. "
            f"Gecerli secenekler: {sorted(allowed)}."
        )

    effective_method = method
    if method == "arima" and period is not None:
        effective_method = "sarimax"
        warnings.warn(
            "Periyot verildi, SARIMAX kullaniliyor.", UserWarning, stacklevel=2
        )

    # ── Sütun doğrulama ──
    if col not in df.columns:
        return {"error": f"Sutun '{col}' DataFrame icinde bulunamadi."}
    series = df[col]
    if not pd.api.types.is_numeric_dtype(series):
        return {"error": f"'{col}' sutunu sayisal degil."}
    series = series.dropna()
    n = len(series)
    if n < 10:
        return {
            "error": (
                f"Seri cok kisa: {n} gozlem (en az 10 gerekir)."
            )
        }

    # ── SARIMAX için periyot zorunluluğu ──
    if effective_method == "sarimax":
        if period is None:
            period = _guess_period(series)
            if period is None:
                return {
                    "error": "SARIMAX icin periyot belirlenemedi; "
                             "lutfen period parametresi verin."
                }
        if n < 2 * period:
            return {
                "error": (
                    f"Seri cok kisa: {n} gozlem, "
                    f"ancak 2*period={2 * period} gerekiyor."
                ),
                "period": period,
            }

    # ── Model eğitimi ve tahmin ──
    # DatetimeIndex yoksa ARIMA ham değerler üzerinde fit edilir → zamansal yapı
    # kaybolur, tahmin istatistiksel olarak anlamsız olabilir. Hesabı engelleme,
    # sadece uyar (MF-1 / ora-9): tüketici (CLI/dashboard) bunu göstermeli.
    has_datetime_index = isinstance(series.index, pd.DatetimeIndex)
    try:
        if effective_method == "arima":
            from statsmodels.tsa.arima.model import ARIMA

            model = ARIMA(series.values, order=(1, 0, 1))
            fitted = model.fit()
            order_repr = "(1,0,1)"
        else:  # sarimax
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            model = SARIMAX(
                series.values,
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, period),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
            order_repr = f"(1,0,1)x(1,0,1,{period})"

        fc = fitted.get_forecast(steps=horizon)
        frame = fc.summary_frame()

        point = frame["mean"].tolist()
        ci_lower = frame["mean_ci_lower"].tolist()
        ci_upper = frame["mean_ci_upper"].tolist()

        return {
            "point": [float(v) for v in point],
            "ci_lower": [float(v) for v in ci_lower],
            "ci_upper": [float(v) for v in ci_upper],
            "method": effective_method,
            "horizon": horizon,
            "order": order_repr,
            "warning": (
                None if has_datetime_index
                else "DatetimeIndex yok; tahmin zamansal yapi olmadan uretiliyor "
                     "ve istatistiksel olarak anlamsiz olabilir."
            ),
        }
    except Exception as e:
        return {"error": f"Model egitimi basarisiz: {e}"}


# ── Yardımcı ──────────────────────────────────────────────────────────────────


def _guess_period(series: pd.Series) -> int | None:
    """Serinin indeks yapısına bakarak muhtemel bir mevsimsellik periyodu döndür.

    - DatetimeIndex + günlük → 7 (haftalık)
    - DatetimeIndex + aylık → 12 (yıllık)
    - DatetimeIndex + çeyreklik → 4 (yıllık)
    - Diğer → None
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        return None

    try:
        freq = pd.infer_freq(series.index)
    except Exception:
        freq = None

    if freq is None:
        # Frekans çıkarılamazsa: ardışık farkların en sık değerine bak
        diffs = series.index.to_series().diff().dropna()
        if len(diffs) == 0:
            return None
        median_diff = diffs.median()
        days = median_diff / pd.Timedelta("1D")
        if 0.8 <= days <= 1.2:
            return 7
        if 27 <= days <= 32:
            return 12
        if 85 <= days <= 100:
            return 4
        return None

    freq = freq.upper()
    if freq.startswith("D") or freq.startswith("B"):
        return 7
    if freq.startswith("M") or freq.startswith("MS") or freq.startswith("BM"):
        return 12
    if freq.startswith("Q"):
        return 4
    if freq.startswith("W"):
        return 52  # haftalık → yıllık
    if freq.startswith("H"):
        return 24  # saatlik → günlük
    return None
