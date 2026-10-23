# Test Fixtures

All data in this directory is **synthetic** — generated programmatically for testing purposes.
No real personal or financial data is included.

## Data Files

- `numeric_heavy.csv` — Mostly numeric columns (yas, gelir, kredi_skoru, etc.) with some missing values and outliers.
- `mixed_with_pii.csv` — Mixed data with synthetic PII columns (tc_kimlik, iban, telefon, email) and numeric columns.
- `categorical.csv` — Mostly categorical columns (il, sektor, urun_tipi, kanal, risk_grubu) plus a numeric target.

## PII Vectors

- `vectors/generate.py` — Programmatically generates valid/invalid TCKN, VKN, IBAN, and PAN test vectors.
- `vectors/*_vectors.json` — Generated test vectors with `valid` and `invalid` arrays.
