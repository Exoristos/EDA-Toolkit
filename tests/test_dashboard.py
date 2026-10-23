"""Phase 8 — dashboard smoke tests + pure helper-function unit tests.

Smoke test uses ``streamlit.testing.v1.AppTest`` to assert the app renders
without error and that the sidebar controls (file uploader + config radios)
exist before any analysis is run.

Pure helper functions (``schemas_to_dataframe`` etc.) are tested in isolation
without a Streamlit runtime.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from eda_toolkit.dashboard import app as dashboard_app
from eda_toolkit.dashboard.app import (
    apply_per_column_mask,
    cleaning_log_to_dataframe,
    findings_to_dataframe,
    hypothesis_tests_to_dataframe,
    outliers_to_dataframe,
    schemas_to_dataframe,
)
from eda_toolkit.types import (
    CleaningAction,
    CleaningLevel,
    ColumnSchema,
    MaskMode,
    PiiFinding,
    PiiType,
)

APP_PATH = str(Path(dashboard_app.__file__))


# ── AppTest smoke: render without error, sidebar controls present ─────────────


def test_app_renders_without_error() -> None:
    """The app should boot and show the start placeholder with no exception."""
    at = AppTest(APP_PATH, default_timeout=30).run()
    assert not at.exception, f"App raised an exception:\n{at.exception}"
    # Placeholder shown before any analysis.
    assert at.info, "Expected the start info placeholder before running analysis."


def test_app_has_file_uploader_and_sidebar_controls() -> None:
    """Sidebar should expose the file uploader + the two config radios."""
    at = AppTest(APP_PATH, default_timeout=30).run()
    assert not at.exception

    # File uploader present.
    assert len(at.file_uploader) >= 1, "File uploader not rendered"
    assert at.file_uploader[0].label == "Veri dosyasi"

    # Cleaning-level + PII-mode radios present (labels match the spec).
    radio_labels = {r.label for r in at.radio}
    assert "Temizleme Seviyesi" in radio_labels
    assert "PII Modu" in radio_labels

    # Primary analyze button present (label check is robust across versions).
    assert any("Analiz Et" in b.label for b in at.button), "'Analiz Et' button missing"

    # Mask-mode selectbox is only rendered when PII mode == Maske; default is
    # "Uyar" so the selectbox should NOT be present on first render.
    assert len(at.selectbox) == 0


def test_app_tabs_only_render_after_analysis() -> None:
    """Before running, the main area shows the info placeholder, not tabs."""
    at = AppTest(APP_PATH, default_timeout=30).run()
    assert not at.exception
    # No tabs widget should exist pre-analysis.
    assert len(at.tabs) == 0


EXAMPLE_CSV = str(
    Path(__file__).resolve().parent.parent / "examples" / "mixed_with_pii.csv"
)


def test_app_runs_pipeline_and_renders_all_tabs() -> None:
    """Upload the example CSV, click Analiz Et, and confirm all 8 tabs render.

    This exercises the temp-file pipeline path + every tab renderer through
    the real :mod:`eda_toolkit.pipeline`. Charts are best-effort; the goal is
    "no exception and the tabs widget appears."
    """
    csv_bytes = Path(EXAMPLE_CSV).read_bytes()
    at = AppTest(APP_PATH, default_timeout=120).run()
    assert not at.exception

    # Upload file, switch PII mode to Maske so the masking UI path is covered.
    at.file_uploader[0].set_value(
        (Path(EXAMPLE_CSV).name, csv_bytes, "text/csv")
    ).run()
    assert not at.exception
    at.radio[1].set_value("Maske").run()
    assert not at.exception

    # Click the Analiz Et button.
    btn = next(b for b in at.button if "Analiz Et" in b.label)
    btn.click().run()
    assert not at.exception, f"Pipeline raised:\n{at.exception}"

    # All eight tabs should now be rendered (AppTest flattens tabs into a list).
    assert len(at.tabs) == 8, f"Expected 8 tabs, got {len(at.tabs)}"
    tab_labels = {t.label for t in at.tabs}
    assert tab_labels == {
        "Ozet", "Tanimlayici Istatistik", "Gorseller",
        "Ileri Analiz", "Tahmin", "Modelleme",
        "PII Uyarilari", "Raporlar",
    }
    # Success message confirming analysis completed.
    assert at.success, "Expected a success message after analysis"


def test_pipeline_carrys_unmasked_original_for_pii_preview() -> None:
    """Regression: when pii==MASK, cleaned_df is masked but the dashboard's PII
    preview needs the UNMASKED original.

    Bug (ora-6): the PII tab read result.cleaned_df as "Orijinal", but that is
    the already-masked report_df, so previews showed masked values and
    HASH/TOKENIZE/PSEUDONYMIZE were applied twice. ReportResult.unmasked_cleaned_df
    carries the true original; this test pins the contract.
    """
    from eda_toolkit import pipeline
    from eda_toolkit.loader import load_dataframe
    from eda_toolkit.types import PiiMode, PipelineConfig

    raw = load_dataframe(EXAMPLE_CSV)
    config = PipelineConfig(
        report_formats=[], pii=PiiMode.MASK, mask_mode=MaskMode.PARTIAL, show_last=4,
    )
    result = pipeline.run(EXAMPLE_CSV, config)

    assert result.unmasked_cleaned_df is not None
    assert result.cleaned_df is not None

    # Unmasked original must equal the freshly loaded dataframe (untouched).
    pd.testing.assert_frame_equal(result.unmasked_cleaned_df, raw)

    # Masked cleaned_df must DIFFER from the unmasked original on PII columns.
    # tc_kimlik is detected as TCKN in mixed_with_pii.csv (AC-008).
    assert "tc_kimlik" in raw.columns
    masked_col = result.cleaned_df["tc_kimlik"].astype(str)
    unmasked_col = result.unmasked_cleaned_df["tc_kimlik"].astype(str)
    assert not masked_col.equals(unmasked_col), (
        "cleaned_df must be masked (partial) while unmasked_cleaned_df holds originals"
    )


# ── Helper function unit tests (pure, no Streamlit) ─────────────────────────


def test_schemas_to_dataframe_basic() -> None:
    schemas = [
        ColumnSchema(name="id", dtype="int64", kind="numeric", unique_count=100, missing_count=0),
        ColumnSchema(name="email", dtype="object", kind="text", unique_count=95, missing_count=5),
    ]
    df = schemas_to_dataframe(schemas)
    assert list(df.columns) == ["Sutun", "Tip", "Sinif", "Essiz Deger", "Eksik"]
    assert len(df) == 2
    assert df.iloc[0]["Sutun"] == "id"
    assert df.iloc[0]["Sinif"] == "numeric"
    assert df.iloc[1]["Eksik"] == 5


def test_schemas_to_dataframe_empty() -> None:
    df = schemas_to_dataframe([])
    assert df.empty
    assert list(df.columns) == ["Sutun", "Tip", "Sinif", "Essiz Deger", "Eksik"]


def test_cleaning_log_to_dataframe() -> None:
    log = [
        CleaningAction(
            column="yas", issue="aykiri", level=CleaningLevel.MODERATE,
            applied=True, detail="alt sinir kirildi",
        ),
        CleaningAction(
            column="gelir", issue="eksik", level=CleaningLevel.CONSERVATIVE,
            applied=False, detail="-dokunulmadi-",
        ),
    ]
    df = cleaning_log_to_dataframe(log)
    assert list(df.columns) == ["Sutun", "Sorun", "Seviye", "Uygulandi", "Detay"]
    assert len(df) == 2
    assert df.iloc[0]["Seviye"] == "moderate"
    assert df.iloc[0]["Uygulandi"] == "Evet"
    assert df.iloc[1]["Uygulandi"] == "Hayir"


def test_cleaning_log_to_dataframe_empty() -> None:
    df = cleaning_log_to_dataframe([])
    assert df.empty
    assert "Sutun" in df.columns


def test_outliers_to_dataframe_none_and_empty() -> None:
    assert outliers_to_dataframe(None).empty
    assert outliers_to_dataframe({}).empty


def test_outliers_to_dataframe_with_values() -> None:
    outliers = {
        "yas": {"count": 3, "rate": 0.04, "lower": 10.0, "upper": 90.0},
        "gelir": {"count": 0, "rate": 0.0, "lower": None, "upper": None},
    }
    df = outliers_to_dataframe(outliers)
    assert list(df.columns) == ["Sutun", "Aykiri Adet", "Oran", "Alt Sinir", "Ust Sinir"]
    assert len(df) == 2
    assert df.iloc[0]["Aykiri Adet"] == 3
    assert df.iloc[0]["Alt Sinir"] == "10.00"
    assert df.iloc[1]["Alt Sinir"] == "-"
    assert df.iloc[1]["Ust Sinir"] == "-"


def test_hypothesis_tests_to_dataframe_empty() -> None:
    assert hypothesis_tests_to_dataframe(None).empty
    assert hypothesis_tests_to_dataframe([]).empty


def test_hypothesis_tests_to_dataframe_shapiro_and_pearson() -> None:
    tests = [
        {"test": "shapiro", "column": "yas", "statistic": 0.97, "p_value": 0.02, "is_normal": False},
        {"test": "shapiro", "column": "sabit", "statistic": None, "p_value": None,
         "is_normal": None, "note": "Sabit sutun (varyans yok)"},
        {"test": "pearson", "columns": ["yas", "gelir"], "statistic": 0.5, "p_value": 0.01,
         "significant": True},
        {"test": "pearson", "columns": ["a", "b"], "statistic": None, "p_value": None,
         "significant": None, "note": "Yetersiz ortak gozlem"},
    ]
    df = hypothesis_tests_to_dataframe(tests)
    assert list(df.columns) == ["Test", "Sutun(lar)", "Istatistik", "p-degeri", "Sonuc"]
    assert len(df) == 4
    assert df.iloc[0]["Test"] == "Shapiro-Wilk"
    assert df.iloc[0]["Sutun(lar)"] == "yas"
    assert df.iloc[0]["Sonuc"] == "Normal degil"
    assert "Sabit" in df.iloc[1]["Sonuc"]  # note surfaced
    assert df.iloc[2]["Test"] == "Pearson"
    assert df.iloc[2]["Sutun(lar)"] == "yas, gelir"
    assert df.iloc[2]["Sonuc"] == "Anlamli"
    assert "Yetersiz" in df.iloc[3]["Sonuc"]


def test_findings_to_dataframe_basic_and_empty() -> None:
    assert findings_to_dataframe([]).empty

    findings = [
        PiiFinding(
            column="tc_kimlik", pii_type=PiiType.TCKN,
            confidence=0.95, match_rate=0.9, sample_count=100,
            detection_layer="name+value",
        ),
    ]
    df = findings_to_dataframe(findings)
    assert list(df.columns) == ["Sutun", "Tur", "Guven (%)", "Eslesme (%)", "Katman"]
    assert len(df) == 1
    assert df.iloc[0]["Sutun"] == "tc_kimlik"
    assert df.iloc[0]["Tur"] == "tckn"
    assert df.iloc[0]["Guven (%)"] == 95.0
    assert df.iloc[0]["Eslesme (%)"] == 90.0


def test_apply_per_column_mask_uses_per_column_modes() -> None:
    """Each PII column should be masked with its own selected mode."""
    df = pd.DataFrame({
        "tc": ["10000000146", "10000000210"],
        "email": ["ali@x.com", "veli@y.com"],
        "name": ["Ali Yilmaz", "Veli Kaya"],
        "amount": [100, 200],
    })
    findings = [
        PiiFinding(column="tc", pii_type=PiiType.TCKN, confidence=0.9,
                   match_rate=0.9, sample_count=2, detection_layer="value"),
        PiiFinding(column="email", pii_type=PiiType.EMAIL, confidence=0.9,
                   match_rate=0.9, sample_count=2, detection_layer="value"),
        PiiFinding(column="name", pii_type=PiiType.NAME, confidence=0.6,
                   match_rate=1.0, sample_count=2, detection_layer="name"),
    ]
    out = apply_per_column_mask(
        df, findings,
        column_modes={"tc": MaskMode.FULL, "email": MaskMode.HASH, "name": MaskMode.PARTIAL},
        show_last=4,
    )
    # FULL → redacted for every tc value.
    assert (out["tc"] == "[REDACTED]").all()
    # HASH → 12-char hex digest, deterministic.
    assert out["email"].iloc[0] == out["email"].iloc[0]  # stable
    assert len(out["email"].iloc[0]) == 12
    # PARTIAL with show_last=4 → trailing 4 chars visible ("Ali Yilmaz" -> "******lmaz").
    assert out["name"].iloc[0].endswith("lmaz")
    # Non-PII column untouched.
    assert out["amount"].tolist() == [100, 200]


def test_apply_per_column_mask_falls_back_to_partial_and_skips_missing() -> None:
    df = pd.DataFrame({"email": ["a@x.com"], "iban": ["TR00"]})
    findings = [
        PiiFinding(column="email", pii_type=PiiType.EMAIL, confidence=0.9,
                   match_rate=0.9, sample_count=1, detection_layer="value"),
        PiiFinding(column="missing_col", pii_type=PiiType.PHONE, confidence=0.5,
                   match_rate=0.5, sample_count=1, detection_layer="value"),
    ]
    out = apply_per_column_mask(
        df, findings, column_modes={}, show_last=2,  # default PARTIAL
    )
    assert list(out.columns) == ["email", "iban"]
    assert out["iban"].iloc[0] == "TR00"  # untouched, not in findings-present cols


def test_masked_dataframe_excel_roundtrip_byte_output() -> None:
    """Smoke-check the CSV/Excel bytes paths used in the download buttons."""
    df = pd.DataFrame({"email": ["a@x.com", "b@y.com"], "amount": [1, 2]})
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    assert b"email" in csv_bytes

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    xlsx_bytes = buf.getvalue()
    assert xlsx_bytes[:2] == b"PK"  # xlsx is a zip
