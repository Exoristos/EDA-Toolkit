"""İleri düzey EDA: aykırı değer tespiti, hipotez testleri, kümeleme."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

from eda_toolkit.eda.correlation import correlation_matrix, categorical_summary


def detect_outliers(df: pd.DataFrame) -> dict:
    """IQR yöntemiyle nümerik sütunlarda aykırı değer tespiti.

    Args:
        df: Girdi DataFrame.

    Returns:
        {column: {"count": int, "rate": float, "lower": float, "upper": float}, ...}
        Aykırı değer: Q1 - 1.5*IQR altı veya Q3 + 1.5*IQR üstü.
    """
    result: dict = {}
    numeric_df = df.select_dtypes(include=[np.number])
    n_rows = len(df)

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) < 4:
            # Yetersiz veri
            result[col] = {"count": 0, "rate": 0.0, "lower": None, "upper": None}  # type: ignore[dict-item]
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        if iqr == 0:
            # Sabit veya çok düşük varyans
            result[col] = {"count": 0, "rate": 0.0, "lower": lower, "upper": upper}
            continue

        outlier_mask = (series < lower) | (series > upper)
        count = int(outlier_mask.sum())

        result[col] = {
            "count": count,
            "rate": round(count / len(series), 4) if len(series) > 0 else 0.0,
            "lower": lower,
            "upper": upper,
        }
    return result


def run_hypothesis_tests(df: pd.DataFrame, alpha: float = 0.05) -> list[dict]:
    """Hipotez testleri: Shapiro-Wilk normallik + Pearson korelasyon.

    - Her nümerik sütun için Shapiro-Wilk testi (en az 20 non-null değer gerekir).
    - En yüksek korelasyona sahip ilk 5 sütun çifti için Pearson korelasyon testi.
    - Sabit sütunlar atlanır; hatalar zarifçe yakalanır.

    Args:
        df: Girdi DataFrame.
        alpha: Anlamlılık düzeyi (varsayılan 0.05).

    Returns:
        Her biri bir test sonucu olan sözlük listesi.
    """
    results: list[dict] = []
    numeric_df = df.select_dtypes(include=[np.number])

    # ── Shapiro-Wilk ──
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) < 20:
            continue
        # Sabit sütun kontrolü
        if series.nunique() <= 1:
            results.append({
                "column": col,
                "test": "shapiro",
                "statistic": None,
                "p_value": None,
                "is_normal": None,
                "note": "Sabit sütun (varyans yok)",
            })
            continue

        try:
            stat, p_value = stats.shapiro(series.values)
            results.append({
                "column": col,
                "test": "shapiro",
                "statistic": float(stat),
                "p_value": float(p_value),
                "is_normal": bool(p_value > alpha),
            })
        except Exception:
            results.append({
                "column": col,
                "test": "shapiro",
                "statistic": None,
                "p_value": None,
                "is_normal": None,
                "note": "Test başarısız oldu",
            })

    # ── Pearson korelasyon (top-5 most-correlated pairs) ──
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr(method="pearson")
        # Üst üçgeni al, diyagonali hariç tut
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                col_a = corr.columns[i]
                col_b = corr.columns[j]
                val = corr.iloc[i, j]
                if not np.isnan(val):
                    pairs.append((abs(val), col_a, col_b, val))

        # Mutlak korelasyon değerine göre sırala, top 5
        pairs.sort(key=lambda x: x[0], reverse=True)
        top_pairs = pairs[:5]

        for _, col_a, col_b, _ in top_pairs:
            a = numeric_df[col_a].dropna()
            b = numeric_df[col_b].dropna()
            # Ortak indeksler
            common_idx = a.index.intersection(b.index)
            if len(common_idx) < 3:
                results.append({
                    "columns": [col_a, col_b],
                    "test": "pearson",
                    "statistic": None,
                    "p_value": None,
                    "significant": None,
                    "note": "Yetersiz ortak gözlem",
                })
                continue
            a_vals = a.loc[common_idx].values
            b_vals = b.loc[common_idx].values

            # Sabit sütunları atla
            if np.std(a_vals) == 0 or np.std(b_vals) == 0:
                results.append({
                    "columns": [col_a, col_b],
                    "test": "pearson",
                    "statistic": None,
                    "p_value": None,
                    "significant": None,
                    "note": "Sabit sütun içeriyor",
                })
                continue

            try:
                r, p_val = stats.pearsonr(a_vals, b_vals)
                results.append({
                    "columns": [col_a, col_b],
                    "test": "pearson",
                    "statistic": float(r),
                    "p_value": float(p_val),
                    "significant": bool(p_val < alpha),
                })
            except Exception:
                results.append({
                    "columns": [col_a, col_b],
                    "test": "pearson",
                    "statistic": None,
                    "p_value": None,
                    "significant": None,
                    "note": "Test başarısız oldu",
                })

    return results


def cluster(df: pd.DataFrame, max_k: int = 4) -> dict:
    """Nümerik sütunlar üzerinde KMeans kümeleme ve IsolationForest anomali tespiti.

    Gereksinimler: en az 2 nümerik sütun ve en az 10 satır.
    K seçimi: silhouette skoru [2..max_k] aralığında en iyi k'yi seçer.
    StandardScaler ile ölçeklendirme yapılır.

    Args:
        df: Girdi DataFrame.
        max_k: Denenecek maksimum küme sayısı (varsayılan 4).

    Returns:
        {
            "performed": bool,
            "k": int | None,
            "labels": pd.Series (dtype int, index=df.index),
            "anomaly": pd.Series (dtype bool, index=df.index),
            "silhouette": float | None,
            "reason": str | None,
        }
    """
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df.dropna()

    if numeric_df.shape[1] < 2 or numeric_df.shape[0] < 10:
        return {
            "performed": False,
            "k": None,
            "labels": pd.Series([-1] * len(df), index=df.index, dtype=int),
            "anomaly": pd.Series([False] * len(df), index=df.index, dtype=bool),
            "silhouette": None,
            "reason": (
                "En az 2 nümerik sütun ve 10 satır gerekiyor. "
                f"Mevcut: {numeric_df.shape[1]} nümerik sütun, {numeric_df.shape[0]} geçerli satır."
            ),
        }

    # Standartlaştırma
    try:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric_df)
    except Exception as e:
        return {
            "performed": False,
            "k": None,
            "labels": pd.Series([-1] * len(df), index=df.index, dtype=int),
            "anomaly": pd.Series([False] * len(df), index=df.index, dtype=bool),
            "silhouette": None,
            "reason": f"Ölçeklendirme başarısız: {e}",
        }

    # Sıfır varyans kontrolü
    if np.any(np.all(scaled == scaled[0, :], axis=0)):
        return {
            "performed": False,
            "k": None,
            "labels": pd.Series([-1] * len(df), index=df.index, dtype=int),
            "anomaly": pd.Series([False] * len(df), index=df.index, dtype=bool),
            "silhouette": None,
            "reason": "Sıfır varyanslı sütun var; kümeleme yapılamaz.",
        }

    # Az sayıda benzersiz satır kontrolü
    unique_rows = np.unique(scaled, axis=0).shape[0]
    effective_max_k = min(max_k, unique_rows - 1)
    if effective_max_k < 2:
        k = 2  # fallback
        effective_max_k = 2

    # Silhouette ile en iyi k
    best_k = 2
    best_silhouette = -1.0
    for k_candidate in range(2, effective_max_k + 1):
        try:
            km = KMeans(n_clusters=k_candidate, random_state=42, n_init=10)
            labels = km.fit_predict(scaled)
            if len(set(labels)) < 2:
                # Tüm noktalar aynı kümeye düştü
                continue
            sil = silhouette_score(scaled, labels)
            if sil > best_silhouette:
                best_silhouette = sil
                best_k = k_candidate
        except Exception:
            continue

    # En iyi k ile KMeans
    if best_silhouette <= -1.0:
        # Silhouette hesaplanamadı, k=2 fallback
        best_k = 2
        best_silhouette = None
        try:
            km = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels_arr = km.fit_predict(scaled)
        except Exception as e:
            return {
                "performed": False,
                "k": None,
                "labels": pd.Series([-1] * len(df), index=df.index, dtype=int),
                "anomaly": pd.Series([False] * len(df), index=df.index, dtype=bool),
                "silhouette": None,
                "reason": f"KMeans başarısız: {e}",
            }
    else:
        try:
            km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels_arr = km.fit_predict(scaled)
        except Exception as e:
            return {
                "performed": False,
                "k": None,
                "labels": pd.Series([-1] * len(df), index=df.index, dtype=int),
                "anomaly": pd.Series([False] * len(df), index=df.index, dtype=bool),
                "silhouette": None,
                "reason": f"KMeans başarısız: {e}",
            }

    # IsolationForest anomali tespiti
    try:
        iso = IsolationForest(random_state=42, contamination=0.05)
        anomaly_pred = iso.fit_predict(scaled)  # 1=normal, -1=anomaly
        anomaly_arr = anomaly_pred == -1
    except Exception:
        # Fallback: tüm False
        anomaly_arr = np.zeros(len(scaled), dtype=bool)

    # Tüm indeks için döndür (NaN satırlar -1/False ile etiketlenir)
    full_labels = pd.Series(-1, index=df.index, dtype=int)
    full_anomaly = pd.Series(False, index=df.index, dtype=bool)

    full_labels.loc[numeric_df.index] = labels_arr
    full_anomaly.loc[numeric_df.index] = anomaly_arr

    return {
        "performed": True,
        "k": int(best_k),
        "labels": full_labels,
        "anomaly": full_anomaly,
        "silhouette": float(best_silhouette) if best_silhouette is not None else None,
    }


def run_advanced_eda(df: pd.DataFrame) -> dict:
    """İleri düzey EDA: korelasyon, kategorik özet, aykırı değer, hipotez testleri, kümeleme.

    Bu fonksiyonun döndüğü sözlük, merkezi entegrasyon aşamasında EdaResult'un
    isteğe bağlı alanlarına eşlenmek üzere tasarlanmıştır.

    Args:
        df: Girdi DataFrame.

    Returns:
        {
            "correlation_matrix": pd.DataFrame,
            "categorical_summary": dict,
            "outliers": dict,
            "hypothesis_tests": list[dict],
            "clustering": dict,
        }
    """
    return {
        "correlation_matrix": correlation_matrix(df),
        "categorical_summary": categorical_summary(df),
        "outliers": detect_outliers(df),
        "hypothesis_tests": run_hypothesis_tests(df),
        "clustering": cluster(df),
    }
