"""Tests for pure helpers in run_openai.py."""
from __future__ import annotations

import re

import pytest

from run_openai import (
    Config, ConfigError, OUTPUT_COLUMNS, RESULT_INPUT_COLUMNS, PROVIDER,
    auto_output_path, parse_args, preflight,
)


def _cfg(**overrides) -> Config:
    base = dict(
        input="in.csv", model="gpt-4o-mini", thinking="off",
        reasoning_effort="medium", explain=False, n=10, temperature=1.0,
        concurrency=5, limit=None, output=None, max_tokens=1024, rpm=400,
        question="",
    )
    base.update(overrides)
    return Config(**base)


def test_parse_args_minimum_required():
    cfg = parse_args([
        "--input", "in.csv",
        "--model", "gpt-4o-mini",
    ])
    assert cfg.input == "in.csv"
    assert cfg.model == "gpt-4o-mini"
    assert cfg.thinking == "off"
    assert cfg.reasoning_effort == "medium"
    assert cfg.explain is False
    assert cfg.n == 10
    assert cfg.temperature == 1.0
    assert cfg.concurrency == 5
    assert cfg.limit is None
    assert cfg.output is None
    assert cfg.max_tokens == 1024
    assert cfg.rpm == 400
    assert cfg.question == ""


def test_parse_args_all_flags():
    cfg = parse_args([
        "--input", "in.csv",
        "--model", "o4-mini",
        "--thinking", "on",
        "--reasoning-effort", "high",
        "--explain",
        "--n", "3",
        "--temperature", "0.5",
        "--concurrency", "2",
        "--limit", "5",
        "--output", "out.csv",
        "--max-tokens", "2048",
        "--rpm", "1200",
        "--question", " Was this acceptable behavior? Answer only with Yes or No.",
    ])
    assert cfg.model == "o4-mini"
    assert cfg.thinking == "on"
    assert cfg.reasoning_effort == "high"
    assert cfg.explain is True
    assert cfg.n == 3
    assert cfg.temperature == 0.5
    assert cfg.concurrency == 2
    assert cfg.limit == 5
    assert cfg.output == "out.csv"
    assert cfg.max_tokens == 2048
    assert cfg.rpm == 1200
    assert cfg.question == " Was this acceptable behavior? Answer only with Yes or No."


def test_parse_args_missing_required_input():
    with pytest.raises(SystemExit):
        parse_args(["--model", "gpt-4o-mini"])


def test_parse_args_missing_required_model():
    with pytest.raises(SystemExit):
        parse_args(["--input", "in.csv"])


def test_parse_args_thinking_choices():
    with pytest.raises(SystemExit):
        parse_args(["--input", "in.csv", "--model", "x", "--thinking", "maybe"])


def test_parse_args_reasoning_effort_choices():
    with pytest.raises(SystemExit):
        parse_args(["--input", "in.csv", "--model", "x",
                    "--reasoning-effort", "extreme"])


def test_provider_constant():
    assert PROVIDER == "openai"


def test_output_columns_match_claude_runner():
    """The unified schema invariant: both runners use the same 25 columns."""
    from run_claude import OUTPUT_COLUMNS as claude_cols
    assert OUTPUT_COLUMNS == claude_cols


def test_output_columns_includes_provider():
    assert "provider" in OUTPUT_COLUMNS


def test_output_columns_includes_reasoning_effort():
    assert "reasoning_effort" in OUTPUT_COLUMNS


def test_output_columns_are_unique():
    assert len(OUTPUT_COLUMNS) == len(set(OUTPUT_COLUMNS))


def test_input_columns_match_claude_runner():
    from run_claude import RESULT_INPUT_COLUMNS as claude_inputs
    assert RESULT_INPUT_COLUMNS == claude_inputs


def test_auto_output_path_pattern_no_reasoning():
    cfg = _cfg(model="gpt-4o-mini", thinking="off", n=10, explain=False)
    p = auto_output_path(cfg)
    assert p.parent.name == "results"
    assert re.match(
        r"gpt-4o-mini_reasoning-off_n10_explain-no_\d{8}-\d{6}\.csv",
        p.name,
    )


def test_auto_output_path_pattern_with_reasoning():
    cfg = _cfg(model="o4-mini", thinking="on", reasoning_effort="high",
               n=5, explain=False)
    p = auto_output_path(cfg)
    assert re.match(
        r"o4-mini_reasoning-high_n5_explain-no_\d{8}-\d{6}\.csv",
        p.name,
    )


def test_auto_output_path_explain_yes():
    cfg = _cfg(explain=True)
    assert "explain-yes" in auto_output_path(cfg).name


def test_preflight_missing_api_key_raises(monkeypatch, tmp_input_csv):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = _cfg(input=str(tmp_input_csv))
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        preflight(cfg)


def test_preflight_missing_input_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    cfg = _cfg(input=str(tmp_path / "missing.csv"))
    with pytest.raises(ConfigError, match="input"):
        preflight(cfg)


def test_preflight_warns_unused_reasoning_effort(monkeypatch, tmp_input_csv, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    cfg = _cfg(input=str(tmp_input_csv), thinking="off", reasoning_effort="high")
    preflight(cfg)
    captured = capsys.readouterr()
    assert "reasoning" in (captured.err + captured.out).lower()


def test_preflight_returns_config_unchanged_when_ok(monkeypatch, tmp_input_csv):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    cfg = _cfg(input=str(tmp_input_csv), thinking="off",
               reasoning_effort="medium")  # medium is the default; no warning
    assert preflight(cfg) == cfg
