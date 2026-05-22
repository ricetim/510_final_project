"""Tests for CSV I/O helpers."""
from __future__ import annotations

from pathlib import Path

import pytest

from run_claude import load_input_rows


def test_load_input_rows_basic(tmp_input_csv: Path):
    rows = load_input_rows(str(tmp_input_csv), limit=None)
    assert len(rows) == 3
    assert rows[0]["base_id"] == "PRU-001"
    assert "Answer only with Yes or No." in rows[0]["full_binary_prompt"]
    assert rows[2]["virtue"] == "Justice"


def test_load_input_rows_respects_limit(tmp_input_csv: Path):
    rows = load_input_rows(str(tmp_input_csv), limit=2)
    assert len(rows) == 2
    assert rows[1]["base_id"] == "PRU-001"


def test_load_input_rows_missing_column_raises(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    bad.write_text("base_id,full_binary_prompt\nX,hello\n")
    with pytest.raises(ValueError, match="missing required column"):
        load_input_rows(str(bad), limit=None)


def test_load_input_rows_empty_raises(tmp_path: Path):
    bad = tmp_path / "empty.csv"
    bad.write_text("base_id,original_scenario_a,variant_description,full_binary_prompt,virtue\n")
    with pytest.raises(ValueError, match="no data rows"):
        load_input_rows(str(bad), limit=None)


def test_load_input_rows_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_input_rows(str(tmp_path / "does-not-exist.csv"), limit=None)
