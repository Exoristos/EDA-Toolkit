from __future__ import annotations

import logging
import os
from pathlib import Path

from eda_toolkit.types import (
    ReportResult,
    PipelineConfig,
    PiiMode,
    MaskMode,
)
from eda_toolkit.loader import load_dataframe
from eda_toolkit.eda.basic import run_basic_eda
from eda_toolkit.eda.advanced import run_advanced_eda
from eda_toolkit.cleaning import clean
from eda_toolkit.pii.detector import detect_pii
from eda_toolkit.pii.masker import mask_dataframe
from eda_toolkit.reporters.html_reporter import write_html
from eda_toolkit.reporters.excel_reporter import render_excel
from eda_toolkit.reporters.pdf_reporter import write_pdf
from eda_toolkit.reporters.docx_reporter import render_docx
from eda_toolkit.reporters.pptx_reporter import render_pptx
from eda_toolkit.analytics.quality import data_quality_score
from eda_toolkit.analytics.risk import pii_risk_score


def run(input_path: str, config: PipelineConfig | None = None) -> ReportResult:
    """Orchestrate the full EDA pipeline.

    1. Load data
    2. Clean (conservative = no-op by default; EDA then runs on the cleaned df)
    3. Run basic EDA on the cleaned dataframe
    4. Run advanced EDA (correlation, outliers, hypothesis tests, clustering)
    5. Detect PII on the real (unmasked) cleaned dataframe
    6. Build ReportResult (masking applied for report_df if PiiMode.MASK)
    7. Produce reports for each format in config.report_formats
    """
    if config is None:
        config = PipelineConfig()

    # 1. Load
    df = load_dataframe(input_path)

    # 2. Clean first, so EDA reflects post-cleaning state (Option B per oracle).
    cleaned_df, cleaning_log = clean(df, config.cleaning)

    # 3. Basic EDA on the cleaned dataframe (schema detected inside basic EDA)
    eda_result = run_basic_eda(cleaned_df)

    # 4. Advanced EDA — populate optional EdaResult fields
    adv = run_advanced_eda(cleaned_df)
    eda_result.correlation_matrix = adv.get("correlation_matrix")
    eda_result.outliers = adv.get("outliers")
    eda_result.hypothesis_tests = adv.get("hypothesis_tests")
    if adv.get("clustering", {}).get("performed"):
        eda_result.cluster_labels = adv["clustering"]["labels"]

    # 5. PII detection on the real (unmasked) cleaned dataframe
    pii_findings = detect_pii(cleaned_df)

    # 5b. Faz-3 analytics that need no user params (defensive — never break the
    # pipeline on a calculation error; the fields stay None on failure).
    try:
        eda_result.quality_score = data_quality_score(cleaned_df)
    except Exception as e:  # noqa: BLE001
        logging.debug("data_quality_score failed: %s", e)
    try:
        risk_mask = config.mask_mode.value if config.pii == PiiMode.MASK else None
        eda_result.risk_score = pii_risk_score(
            cleaned_df, pii_findings, mask_mode=risk_mask
        )
    except Exception as e:  # noqa: BLE001
        logging.debug("pii_risk_score failed: %s", e)

    # 6. Build report_df: if MASK mode, mask PII columns; otherwise use original
    if config.pii == PiiMode.MASK and pii_findings:
        report_df = mask_dataframe(
            cleaned_df, pii_findings, config.mask_mode,
            show_last=config.show_last,
        )
    else:
        report_df = cleaned_df

    # 7. Build ReportResult
    report_result = ReportResult(
        input_path=input_path,
        row_count=len(df),
        col_count=len(df.columns),
        cleaned_df=report_df,
        eda=eda_result,
        pii_findings=pii_findings,
        cleaning_log=cleaning_log,
        reports={},
        unmasked_cleaned_df=cleaned_df,
    )

    # 8. Produce reports
    output_dir = Path("eda_reports")
    output_dir.mkdir(exist_ok=True)

    basename = Path(input_path).stem
    for fmt in config.report_formats:
        if fmt == "html":
            out_path = str(output_dir / f"{basename}_report.html")
            write_html(report_result, config, out_path)
            report_result.reports["html"] = out_path
        elif fmt == "excel":
            out_path = str(output_dir / f"{basename}_report.xlsx")
            render_excel(report_result, config, out_path)
            report_result.reports["excel"] = out_path
        elif fmt == "pdf":
            out_path = str(output_dir / f"{basename}_report.pdf")
            write_pdf(report_result, config, out_path)
            report_result.reports["pdf"] = out_path
        elif fmt == "docx":
            out_path = str(output_dir / f"{basename}_report.docx")
            render_docx(report_result, config, out_path)
            report_result.reports["docx"] = out_path
        elif fmt == "pptx":
            out_path = str(output_dir / f"{basename}_report.pptx")
            render_pptx(report_result, config, out_path)
            report_result.reports["pptx"] = out_path
        else:
            print(f"Bilinmeyen rapor formati atlandi: {fmt}")

    return report_result
