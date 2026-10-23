from __future__ import annotations

import warnings

import pandas as pd
import requests

from eda_toolkit.loader import load_dataframe, load_many, load_from_url


# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_csv(path: str, content: str) -> None:
    """Write a CSV string to a file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ── FakeResponse for monkeypatching requests.get ─────────────────────────────

class FakeResponse:
    """Minimal mock for requests.Response."""
    def __init__(self, status_code: int, content: bytes):
        self.status_code = status_code
        self.content = content


# ── load_many: concat mode ───────────────────────────────────────────────────

def test_concat_same_schema(tmp_path):
    """Two same-schema CSVs → row count is sum of individual row counts."""
    csv1 = tmp_path / "a.csv"
    csv2 = tmp_path / "b.csv"
    _write_csv(str(csv1), "x,y\n1,10\n2,20\n")
    _write_csv(str(csv2), "x,y\n3,30\n4,40\n5,50\n")

    result = load_many([str(csv1), str(csv2)], mode="concat")
    assert len(result) == 5
    assert list(result.columns) == ["x", "y"]
    assert result["x"].tolist() == [1, 2, 3, 4, 5]


def test_concat_extra_column(tmp_path):
    """One file has an extra column → union columns + warning."""
    csv1 = tmp_path / "a.csv"
    csv2 = tmp_path / "b.csv"
    _write_csv(str(csv1), "x,y\n1,10\n2,20\n")
    _write_csv(str(csv2), "x,y,z\n3,30,100\n4,40,200\n")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = load_many([str(csv1), str(csv2)], mode="concat")

    # Check warning was emitted for the column count mismatch
    warn_messages = [str(m.message) for m in w]
    assert any("sutun sayisi farkli" in msg for msg in warn_messages)

    # All three columns should appear
    assert set(result.columns) == {"x", "y", "z"}
    # Rows from csv1 should have NaN in z
    assert pd.isna(result.loc[0, "z"])
    assert pd.isna(result.loc[1, "z"])
    # Rows from csv2 should have z values
    assert result.loc[2, "z"] == 100
    assert result.loc[3, "z"] == 200


def test_concat_empty_paths():
    """Empty paths list raises ValueError."""
    import pytest
    with pytest.raises(ValueError, match="Path listesi bos olamaz"):
        load_many([], mode="concat")


# ── load_many: join mode ─────────────────────────────────────────────────────

def test_join_on_key(tmp_path):
    """Join two CSVs on a common key → merged columns from both files."""
    csv1 = tmp_path / "a.csv"
    csv2 = tmp_path / "b.csv"
    _write_csv(str(csv1), "id,name\n1,Alice\n2,Bob\n")
    _write_csv(str(csv2), "id,score\n1,95\n2,88\n3,76\n")

    result = load_many([str(csv1), str(csv2)], mode="join", on="id")
    # Left join: should have rows from csv1 only (2 rows)
    assert len(result) == 2
    assert set(result.columns) == {"id", "name", "score"}
    assert result.loc[0, "name"] == "Alice"
    assert result.loc[0, "score"] == 95


def test_join_missing_key(tmp_path):
    """Missing join key in subsequent file raises ValueError."""
    csv1 = tmp_path / "a.csv"
    csv2 = tmp_path / "b.csv"
    _write_csv(str(csv1), "id,name\n1,Alice\n")
    _write_csv(str(csv2), "other,score\n1,95\n")

    import pytest
    with pytest.raises(ValueError, match="Join anahtari"):
        load_many([str(csv1), str(csv2)], mode="join", on="id")


def test_join_missing_on_param(tmp_path):
    """Join mode without on parameter raises ValueError."""
    csv1 = tmp_path / "a.csv"
    _write_csv(str(csv1), "id,name\n1,Alice\n")

    import pytest
    with pytest.raises(ValueError, match="'on' parametresi zorunludur"):
        load_many([str(csv1)], mode="join")


# ── load_from_url ────────────────────────────────────────────────────────────

def test_load_from_url_csv_success(monkeypatch):
    """load_from_url parses CSV content returned by mocked requests.get."""

    def mock_get(url, headers=None, timeout=30, verify=True):
        return FakeResponse(status_code=200, content=b"a,b\n1,2\n3,4\n")

    monkeypatch.setattr("eda_toolkit.loader.requests.get", mock_get)

    df = load_from_url("https://example.com/data.csv", fmt="csv")
    assert len(df) == 2
    assert list(df.columns) == ["a", "b"]
    assert df["a"].tolist() == [1, 3]
    assert df["b"].tolist() == [2, 4]


def test_load_from_url_status_500(monkeypatch):
    """HTTP 500 status raises RuntimeError."""

    def mock_get(url, headers=None, timeout=30, verify=True):
        return FakeResponse(status_code=500, content=b"Internal Server Error")

    monkeypatch.setattr("eda_toolkit.loader.requests.get", mock_get)

    import pytest
    with pytest.raises(RuntimeError, match="URL hatasi 500"):
        load_from_url("https://example.com/bad.csv")


def test_load_from_url_timeout(monkeypatch):
    """Request timeout raises RuntimeError with Turkish timeout message."""

    def mock_get(url, headers=None, timeout=30, verify=True):
        raise requests.exceptions.Timeout("Connection timed out")

    monkeypatch.setattr("eda_toolkit.loader.requests.get", mock_get)

    import pytest
    with pytest.raises(RuntimeError, match="Zaman asimi"):
        load_from_url("https://example.com/slow.csv")


def test_load_from_url_invalid_format(monkeypatch):
    """Unknown format raises ValueError."""

    def mock_get(url, headers=None, timeout=30, verify=True):
        return FakeResponse(status_code=200, content=b"dummy")

    monkeypatch.setattr("eda_toolkit.loader.requests.get", mock_get)

    import pytest
    with pytest.raises(ValueError, match="Desteklenmeyen format"):
        load_from_url("https://example.com/data.xyz", fmt="xyz")


def test_load_from_url_json(monkeypatch):
    """load_from_url with fmt='json' parses JSON correctly."""

    def mock_get(url, headers=None, timeout=30, verify=True):
        return FakeResponse(status_code=200, content=b'[{"x":1,"y":2},{"x":3,"y":4}]')

    monkeypatch.setattr("eda_toolkit.loader.requests.get", mock_get)

    df = load_from_url("https://example.com/data.json", fmt="json")
    assert len(df) == 2
    assert list(df.columns) == ["x", "y"]


# ── Import smoke test ────────────────────────────────────────────────────────

def test_imports():
    """Verify load_many and load_from_url are importable."""
    from eda_toolkit.loader import load_many, load_from_url
    assert callable(load_many)
    assert callable(load_from_url)
