from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd


class CleaningLevel(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class PiiMode(str, Enum):
    WARN = "warn"
    MASK = "mask"


class MaskMode(str, Enum):
    PARTIAL = "partial"
    FULL = "full"
    HASH = "hash"
    TOKENIZE = "tokenize"
    PSEUDONYMIZE = "pseudonymize"


class PiiType(str, Enum):
    TCKN = "tckn"
    VKN = "vkn"
    IBAN = "iban"
    PHONE = "phone"
    EMAIL = "email"
    CARD_PAN = "card_pan"
    NAME = "name"
    DOB = "dob"
    ADDRESS = "address"


@dataclass
class ColumnSchema:
    name: str
    dtype: str
    kind: str
    unique_count: int
    missing_count: int


@dataclass
class PiiFinding:
    column: str
    pii_type: PiiType
    confidence: float
    match_rate: float
    sample_count: int
    detection_layer: str


@dataclass
class CleaningAction:
    column: str
    issue: str
    level: CleaningLevel
    applied: bool
    detail: str


@dataclass
class EdaResult:
    descriptive: pd.DataFrame
    missing_summary: pd.DataFrame
    schemas: list[ColumnSchema]
    # EXTENSION PROTOCOL: advanced-EDA (Phase 3) MUST add new fields here as
    # Optional with None defaults so basic-EDA tests and the HTML reporter keep
    # working without changes. Do NOT add required positional fields.
    correlation_matrix: pd.DataFrame | None = None
    outliers: dict | None = None
    hypothesis_tests: list | None = None
    cluster_labels: pd.Series | None = None
    # Faz-3 analytics (optional, None defaults — Faz-1 extension protocol; dict
    # returns per ora-8 NH-1, NOT dataclasses). Populated defensively by pipeline.
    timeseries: dict | None = None
    regression: dict | None = None
    quality_score: dict | None = None
    risk_score: dict | None = None


@dataclass
class ReportResult:
    input_path: str
    row_count: int
    col_count: int
    cleaned_df: "pd.DataFrame | None"
    eda: EdaResult
    pii_findings: list[PiiFinding]
    cleaning_log: list[CleaningAction]
    reports: dict = field(default_factory=dict)  # format -> output path
    # Faz-2 dashboard: original (unmasked) dataframe for the PII-tab masking
    # preview. When pii==MASK, cleaned_df carries the masked report_df and the
    # dashboard needs the unmasked original for side-by-side preview + correct
    # interactive masking (HASH/TOKENIZE/PSEUDONYMIZE must not be applied twice).
    # Optional + None default so CLI and reporters are unaffected (Faz-1 protocol).
    unmasked_cleaned_df: "pd.DataFrame | None" = None


@dataclass
class PipelineConfig:
    report_formats: list = field(default_factory=lambda: ["html"])
    cleaning: CleaningLevel = CleaningLevel.CONSERVATIVE
    pii: PiiMode = PiiMode.WARN
    mask_mode: MaskMode = MaskMode.PARTIAL
    show_last: int = 4
    lang: str = "tr"
