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


from run_claude import parse_response


def test_parse_response_tool_use(tool_use_response):
    r = tool_use_response(answer="Yes", explanation="ok",
                           input_tokens=120, output_tokens=15)
    parsed = parse_response(r, latency_ms=234)
    assert parsed["answer"] == "Yes"
    assert parsed["explanation"] == "ok"
    assert parsed["stop_reason"] == "tool_use"
    assert parsed["input_tokens"] == 120
    assert parsed["output_tokens"] == 15
    assert parsed["cache_read_tokens"] == 0
    assert parsed["cache_creation_tokens"] == 0
    assert parsed["latency_ms"] == 234
    assert parsed["error"] == ""


def test_parse_response_text_only_no_tool_call(text_only_response):
    r = text_only_response(text="Hmm", stop_reason="end_turn")
    parsed = parse_response(r, latency_ms=50)
    assert parsed["answer"] == ""
    assert parsed["explanation"] == ""
    assert parsed["stop_reason"] == "end_turn"
    assert "no tool_use block" in parsed["error"].lower()


def test_parse_response_missing_explanation(tool_use_response):
    """If the model omits explanation, store empty string, not None."""
    r = tool_use_response(answer="No", explanation=None)
    # Simulate model omitting the key entirely
    r.content[0].input = {"answer": "No"}
    parsed = parse_response(r, latency_ms=10)
    assert parsed["answer"] == "No"
    assert parsed["explanation"] == ""
