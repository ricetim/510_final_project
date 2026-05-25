"""Tests for pure helpers in run_claude_text.py."""
from __future__ import annotations

import re

import pytest

from run_claude_text import (
    Config, ConfigError, OUTPUT_COLUMNS, RESULT_INPUT_COLUMNS, PROVIDER,
    auto_output_path, parse_args, preflight,
)


def _cfg(**overrides) -> Config:
    base = dict(
        input="in.csv", model="claude-haiku-4-5-20251001", thinking="off",
        thinking_budget=4096, explain=False, n=10, temperature=1.0,
        concurrency=5, limit=None, output=None, max_tokens=1024, rpm=45,
        question="",
    )
    base.update(overrides)
    return Config(**base)


def test_parse_args_minimum_required():
    cfg = parse_args([
        "--input", "in.csv",
        "--model", "claude-haiku-4-5-20251001",
    ])
    assert cfg.input == "in.csv"
    assert cfg.model == "claude-haiku-4-5-20251001"
    assert cfg.thinking == "off"
    assert cfg.thinking_budget == 4096
    assert cfg.explain is False
    assert cfg.n == 10
    assert cfg.temperature == 1.0
    assert cfg.concurrency == 5
    assert cfg.limit is None
    assert cfg.output is None
    assert cfg.max_tokens == 1024
    assert cfg.rpm == 45
    assert cfg.question == ""


def test_parse_args_all_flags():
    cfg = parse_args([
        "--input", "in.csv",
        "--model", "claude-sonnet-4-6",
        "--thinking", "on",
        "--thinking-budget", "8000",
        "--n", "3",
        "--temperature", "0.5",
        "--concurrency", "2",
        "--limit", "5",
        "--output", "out.csv",
        "--max-tokens", "2048",
        "--rpm", "120",
        "--question", " Was this acceptable behavior? Answer only with Yes or No.",
    ])
    assert cfg.model == "claude-sonnet-4-6"
    assert cfg.thinking == "on"
    assert cfg.thinking_budget == 8000
    assert cfg.n == 3
    assert cfg.temperature == 0.5
    assert cfg.concurrency == 2
    assert cfg.limit == 5
    assert cfg.output == "out.csv"
    assert cfg.max_tokens == 2048
    assert cfg.rpm == 120
    assert cfg.question == " Was this acceptable behavior? Answer only with Yes or No."


def test_parse_args_missing_required_input():
    with pytest.raises(SystemExit):
        parse_args(["--model", "claude-haiku-4-5-20251001"])


def test_parse_args_missing_required_model():
    with pytest.raises(SystemExit):
        parse_args(["--input", "in.csv"])


def test_provider_constant():
    assert PROVIDER == "anthropic"


def test_output_columns_match_claude_runner():
    """Same 25-column schema as the other runners."""
    from run_claude import OUTPUT_COLUMNS as claude_cols
    assert OUTPUT_COLUMNS == claude_cols


def test_input_columns_match_claude_runner():
    from run_claude import RESULT_INPUT_COLUMNS as claude_inputs
    assert RESULT_INPUT_COLUMNS == claude_inputs


def test_auto_output_path_has_text_infix():
    cfg = _cfg(model="claude-haiku-4-5-20251001", thinking="off", n=10)
    p = auto_output_path(cfg)
    assert p.parent.name == "results"
    assert re.match(
        r"claude-haiku-4-5-20251001_text_think-off_n10_explain-no_\d{8}-\d{6}\.csv",
        p.name,
    )


def test_auto_output_path_pattern_with_thinking():
    cfg = _cfg(model="claude-sonnet-4-6", thinking="on", n=5)
    p = auto_output_path(cfg)
    assert "_text_think-on_" in p.name


def test_preflight_missing_api_key_raises(monkeypatch, tmp_input_csv):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = _cfg(input=str(tmp_input_csv))
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        preflight(cfg)


def test_preflight_missing_input_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = _cfg(input=str(tmp_path / "missing.csv"))
    with pytest.raises(ConfigError, match="input"):
        preflight(cfg)


def test_preflight_explain_raises(monkeypatch, tmp_input_csv):
    """text mode cannot enforce structured explanations — preflight rejects --explain."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = _cfg(input=str(tmp_input_csv), explain=True)
    with pytest.raises(ConfigError, match="text mode does not support --explain"):
        preflight(cfg)


def test_preflight_temperature_forced_when_thinking_on(monkeypatch, tmp_input_csv, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = _cfg(input=str(tmp_input_csv), thinking="on", temperature=0.5)
    out = preflight(cfg)
    assert out.temperature == 1.0
    captured = capsys.readouterr()
    assert "temperature" in (captured.err + captured.out).lower()


def test_preflight_warns_unused_budget(monkeypatch, tmp_input_csv, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = _cfg(input=str(tmp_input_csv), thinking="off", thinking_budget=9000)
    preflight(cfg)
    captured = capsys.readouterr()
    assert "ignored" in (captured.err + captured.out).lower()


def test_preflight_returns_config_unchanged_when_ok(monkeypatch, tmp_input_csv):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = _cfg(input=str(tmp_input_csv), thinking="off")
    assert preflight(cfg) == cfg
