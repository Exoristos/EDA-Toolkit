from __future__ import annotations

from pathlib import Path
import pytest

FIXTURES_DATA = Path(__file__).parent / "fixtures" / "data"


@pytest.fixture
def numeric_heavy_csv() -> str:
    return str(FIXTURES_DATA / "numeric_heavy.csv")


@pytest.fixture
def mixed_with_pii_csv() -> str:
    return str(FIXTURES_DATA / "mixed_with_pii.csv")


@pytest.fixture
def categorical_csv() -> str:
    return str(FIXTURES_DATA / "categorical.csv")
