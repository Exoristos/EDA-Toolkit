"""Turkish PII detection and masking module.

Public API:
    detect_pii(df, sample_size=100, threshold=0.8) -> list[PiiFinding]
    mask_value(value, pii_type, mode, show_last=4, salt=None) -> str | None
    mask_dataframe(df, findings, mode, show_last=4, salt=None) -> pd.DataFrame
    is_valid_tckn(value) -> bool
    is_valid_vkn(value) -> bool
    is_valid_iban(value) -> bool
    is_valid_phone(value) -> bool
    is_valid_email(value) -> bool
    is_valid_card_pan(value) -> bool
    match_rate(values, validator) -> float
"""

from eda_toolkit.pii.validators import (
    is_valid_tckn,
    is_valid_vkn,
    is_valid_iban,
    is_valid_phone,
    is_valid_email,
    is_valid_card_pan,
    match_rate,
)
from eda_toolkit.pii.detector import detect_pii
from eda_toolkit.pii.masker import mask_value, mask_dataframe

__all__ = [
    "detect_pii",
    "mask_value",
    "mask_dataframe",
    "is_valid_tckn",
    "is_valid_vkn",
    "is_valid_iban",
    "is_valid_phone",
    "is_valid_email",
    "is_valid_card_pan",
    "match_rate",
]
