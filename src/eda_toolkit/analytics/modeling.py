"""Regresyon modelleme: dogrusal, lojistik, VIF hesaplama.

Her fonksiyon dict dondurur (ora-8 NH-1). Statsmodels ic aktarmalari
fonksiyon icinde yapilir (lazy import).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_regression(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    kind: str = "linear",
) -> dict:
    """Regresyon modeli fit eder; katsayi, p-degeri, R², VIF dondurur.

    Args:
        df: Girdi DataFrame.
        target: Hedef sutun adi.
        features: Bagimsiz degisken sutun adlari.
        kind: "linear" (dogrusal) veya "logistic" (ikili siniflandirma).

    Returns:
        dict: model ozeti.
    """
    try:
        if kind == "linear":
            return _fit_linear(df, target, features)
        elif kind == "logistic":
            return _fit_logistic(df, target, features)
        else:
            return {"error": f"Bilinmeyen kind: {kind!r}. 'linear' veya 'logistic' bekleniyor."}  # noqa: TRY003
    except Exception as exc:
        return {"error": str(exc)}


# ── Linear regression ──────────────────────────────────────────────────────────

def _fit_linear(df: pd.DataFrame, target: str, features: list[str]) -> dict:
    """Dogrusal regresyon (statsmodels OLS)."""
    # Temizlik
    X_raw, y, feat_names = _prepare_design_matrix(df, target, features)

    n_required = len(features) + 1
    if len(X_raw) < n_required:
        return {"error": f"Yetersiz satir: {len(X_raw)} gozlem, {len(features)} ozellik. En az {n_required} gerekir."}

    import statsmodels.api as sm

    X = sm.add_constant(X_raw, has_constant="add")
    model = sm.OLS(y, X).fit()

    param_order = ["intercept"] + feat_names
    coef_map = dict(zip(param_order, model.params.tolist(), strict=False))
    pval_map = dict(zip(param_order, model.pvalues.tolist(), strict=False))

    # VIF
    vif_map, vif_warn = _compute_vif(X_raw, feat_names)

    return {
        "kind": "linear",
        "r2": float(model.rsquared),
        "coefficients": coef_map,
        "p_values": pval_map,
        "vif": vif_map,
        "vif_warning": vif_warn,
        "n": int(len(X_raw)),
        "intercept": float(model.params[0]),  # sabit terim ilk sirada
        "feature_names": feat_names,
    }


# ── Logistic regression ────────────────────────────────────────────────────────

def _fit_logistic(df: pd.DataFrame, target: str, features: list[str]) -> dict:
    """Lojistik regresyon (statsmodels Logit; fallback sklearn LogisticRegression)."""
    X_raw, y, feat_names = _prepare_design_matrix(df, target, features)

    n_required = len(features) + 1
    if len(X_raw) < n_required:
        return {"error": f"Yetersiz satir: {len(X_raw)} gozlem, {len(features)} ozellik. En az {n_required} gerekir."}

    # VIF
    vif_map, vif_warn = _compute_vif(X_raw, feat_names)

    result: dict = {
        "kind": "logistic",
        "vif": vif_map,
        "vif_warning": vif_warn,
        "n": int(len(X_raw)),
        "feature_names": feat_names,
    }

    sm_ok = False
    try:
        import statsmodels.api as sm

        X_sm = sm.add_constant(X_raw, has_constant="add")
        logit_model = sm.Logit(y, X_sm)
        fit_result = logit_model.fit(disp=0)

        param_order = ["intercept"] + feat_names
        coef_map = dict(zip(param_order, fit_result.params.tolist(), strict=False))
        pval_map = dict(zip(param_order, fit_result.pvalues.tolist(), strict=False))

        result["pseudo_r2"] = float(fit_result.prsquared)
        result["coefficients"] = coef_map
        result["p_values"] = pval_map
        result["intercept"] = float(fit_result.params[0])
        sm_ok = True
    except Exception:
        pass

    if not sm_ok:
        # Fallback: sklearn LogisticRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        clf = LogisticRegression(max_iter=2000, random_state=0)
        clf.fit(X_raw, y)
        y_pred = clf.predict(X_raw)
        coef_map = dict(zip(feat_names, clf.coef_[0].tolist(), strict=False))
        coef_map["intercept"] = float(clf.intercept_[0])

        result["accuracy"] = float(accuracy_score(y, y_pred))
        result["coefficients"] = coef_map
        result["intercept"] = float(clf.intercept_[0])
        # No p_values from sklearn fallback

    return result


# ── Yardimcilar ────────────────────────────────────────────────────────────────

def _prepare_design_matrix(
    df: pd.DataFrame,
    target: str,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Hedef ve ozellik sutunlarini secer, NaN'leri temizler, one-hot açar.

    Returns:
        (X, y, feature_names): numpy dizileri ve ozellik adlari.
    """
    all_cols = [target] + features
    sub = df[all_cols].dropna().reset_index(drop=True)
    y = sub[target].values

    # One-hot encode non-numeric features
    X_df = sub[features]
    numeric_mask = X_df.select_dtypes(include=[np.number]).columns
    object_cols = [c for c in X_df.columns if c not in numeric_mask]

    if object_cols:
        X_num = X_df[numeric_mask].copy() if len(numeric_mask) > 0 else pd.DataFrame()
        X_cat = pd.get_dummies(X_df[object_cols], drop_first=True).astype(float)
        X_df = pd.concat([X_num, X_cat], axis=1)
    else:
        X_df = X_df.astype(float)

    X = X_df.values.astype(float)
    feat_names = X_df.columns.tolist()
    return X, y, feat_names


def _compute_vif(
    X: np.ndarray,
    feat_names: list[str],
) -> tuple[dict, list[str]]:
    """VIF hesaplar; VIF > 10 olanlari uyari listesine ekler."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_map: dict[str, float] = {}
    vif_warn: list[str] = []

    # VIF requires at least 2 features
    n_cols = X.shape[1]
    if n_cols < 2:
        for i, name in enumerate(feat_names):
            vif_map[name] = float("nan")
        return vif_map, vif_warn

    for i, name in enumerate(feat_names):
        try:
            vif_val = float(variance_inflation_factor(X, i))
        except np.linalg.LinAlgError:
            vif_val = float("inf")
        except Exception:
            vif_val = float("nan")

        vif_map[name] = vif_val
        if vif_val > 10 and not np.isnan(vif_val):
            vif_warn.append(name)

    return vif_map, vif_warn
