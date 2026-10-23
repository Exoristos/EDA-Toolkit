# EDA Toolkit — Privacy-Aware Data Quality Pipeline

A portable Python toolkit for profiling, cleaning and reporting tabular data.
It is designed for analyst workflows where data quality and Turkish personal
data detection matter as much as charts and summary statistics.

## Why this project exists

Analysts often repeat the same first steps for every CSV or Excel file: inspect
types, find missing values, detect outliers, check sensitive identifiers and
prepare a report. EDA Toolkit turns those steps into a consistent command-line
pipeline and produces shareable HTML, Excel, PDF, Word and PowerPoint outputs.

## Capabilities

- CSV, JSON and Excel loading, including multiple-file join/concat workflows
- configurable conservative, moderate and aggressive cleaning
- Turkish TCKN, VKN, IBAN and payment-card checks with masking options
- data quality and re-identification risk scores
- static and interactive visualisations
- HTML, XLSX, PDF, DOCX and PPTX report generation
- ARIMA/SARIMAX time-series forecasting
- linear and logistic regression from the CLI
- automated tests covering cleaning, PII, loaders, modelling and reporters

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .
eda examples/mixed_with_pii.csv --report html,excel --cleaning conservative --pii mask
```

Time-series forecasting:

```bash
eda ts sales.csv --col revenue --horizon 6 --method sarimax --period 12
```

Regression:

```bash
eda model customers.csv --target churn --features recency,frequency,value --kind logistic
```

## Architecture

```text
src/eda_toolkit/
  loader.py       file and URL ingestion
  cleaning.py     typed cleaning policies
  pipeline.py     orchestration and result model
  pii/            identifier detection and masking
  analytics/      quality, risk, modelling and time-series analysis
  reporters/      HTML, Excel, PDF, Word and PowerPoint outputs
  cli.py          Click command-line interface
tests/             unit, integration and format-specific tests
examples/          synthetic demonstration datasets
```

## Portfolio relevance

This repository demonstrates more than exploratory notebooks: it packages an
analyst workflow as reusable software, defines privacy-aware defaults, exposes a
CLI and tests output across multiple business report formats. The project is
especially relevant to banking, insurance, CRM and operational analytics roles.

## Validation

```bash
pip install -r requirements-dev.txt
pytest -q
```

The bundled example data is synthetic. Generated reports and local build
artifacts are excluded from version control.

