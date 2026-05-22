"""Tests for CSV I/O helpers."""
from __future__ import annotations

import csv as _csv
from pathlib import Path

import pytest

from run_claude import OUTPUT_COLUMNS, ResultWriter, load_input_rows


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


def test_result_writer_writes_header(tmp_path):
    out = tmp_path / "out.csv"
    w = ResultWriter(out)
    w.close()
    with out.open() as f:
        header = next(_csv.reader(f))
    assert header == OUTPUT_COLUMNS


def test_result_writer_appends_rows(tmp_path):
    out = tmp_path / "out.csv"
    w = ResultWriter(out)
    row = {c: "" for c in OUTPUT_COLUMNS}
    row.update({"base_id": "X", "answer": "Yes", "explanation": "with, comma"})
    w.write(row)
    w.close()
    with out.open() as f:
        reader = _csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["base_id"] == "X"
    assert rows[0]["answer"] == "Yes"
    assert rows[0]["explanation"] == "with, comma"


def test_result_writer_ignores_extra_keys(tmp_path):
    out = tmp_path / "out.csv"
    w = ResultWriter(out)
    row = {c: "" for c in OUTPUT_COLUMNS}
    row["unknown_key"] = "should not crash"
    w.write(row)
    w.close()
