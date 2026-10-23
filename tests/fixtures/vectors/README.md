# PII Test Vectors

Generated programmatically by `generate.py`. All values are **synthetic** — no real PII data.

- `tckn_vectors.json` — Valid TCKN (including NVI canonical `10000000146`) and invalid entries.
- `vkn_vectors.json` — Valid VKN (including canonical `1000036101`) and invalid entries.
- `iban_vectors.json` — Valid synthetic TR IBANs (MOD-97) and invalid entries.
- `pan_vectors.json` — Valid Luhn 16-digit PANs and invalid entries.

Each JSON file has `{"valid": [...], "invalid": [...]}` structure.
