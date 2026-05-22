"""Tests for pure helpers in run_claude.py."""
from __future__ import annotations

import pytest

from run_claude import Config, parse_args


def test_parse_args_minimum_required():
    cfg = parse_args([
        "--input", "in.csv",
        "--model", "claude-opus-4-7",
    ])
    assert cfg.input == "in.csv"
    assert cfg.model == "claude-opus-4-7"
    assert cfg.thinking == "off"           # default
    assert cfg.thinking_budget == 4096      # default
    assert cfg.explain is False
    assert cfg.n == 10
    assert cfg.temperature == 1.0
    assert cfg.concurrency == 5
    assert cfg.limit is None
    assert cfg.output is None
    assert cfg.max_tokens == 1024


def test_parse_args_all_flags():
    cfg = parse_args([
        "--input", "in.csv",
        "--model", "claude-sonnet-4-6",
        "--thinking", "on",
        "--thinking-budget", "8000",
        "--explain",
        "--n", "3",
        "--temperature", "0.5",
        "--concurrency", "2",
        "--limit", "5",
        "--output", "out.csv",
        "--max-tokens", "2048",
    ])
    assert cfg.model == "claude-sonnet-4-6"
    assert cfg.thinking == "on"
    assert cfg.thinking_budget == 8000
    assert cfg.explain is True
    assert cfg.n == 3
    assert cfg.temperature == 0.5
    assert cfg.concurrency == 2
    assert cfg.limit == 5
    assert cfg.output == "out.csv"
    assert cfg.max_tokens == 2048


def test_parse_args_missing_required_input():
    with pytest.raises(SystemExit):
        parse_args(["--model", "claude-opus-4-7"])


def test_parse_args_missing_required_model():
    with pytest.raises(SystemExit):
        parse_args(["--input", "in.csv"])


def test_parse_args_thinking_choices():
    with pytest.raises(SystemExit):
        parse_args(["--input", "in.csv", "--model", "x", "--thinking", "maybe"])


from run_claude import build_tool_schema


def test_tool_schema_explain_off():
    schema = build_tool_schema(explain=False)
    assert schema["name"] == "submit_answer"
    props = schema["input_schema"]["properties"]
    assert props["answer"]["enum"] == ["Yes", "No"]
    assert "explanation" in props
    assert schema["input_schema"]["required"] == ["answer"]


def test_tool_schema_explain_on():
    schema = build_tool_schema(explain=True)
    assert set(schema["input_schema"]["required"]) == {"answer", "explanation"}


def test_tool_schema_is_independent_per_call():
    """Mutating one returned schema must not affect a later call."""
    a = build_tool_schema(explain=False)
    a["input_schema"]["required"].append("explanation")
    b = build_tool_schema(explain=False)
    assert b["input_schema"]["required"] == ["answer"]
