"""Tests for API call construction and execution (mocked)."""
from __future__ import annotations

import pytest

from run_claude import Config, build_tool_schema, make_request_kwargs


def _cfg(**overrides) -> Config:
    base = dict(
        input="in.csv", model="claude-opus-4-7", thinking="off",
        thinking_budget=4096, explain=False, n=10, temperature=1.0,
        concurrency=5, limit=None, output=None, max_tokens=1024,
    )
    base.update(overrides)
    return Config(**base)


def test_make_request_kwargs_no_thinking():
    cfg = _cfg(thinking="off", model="claude-opus-4-7", temperature=0.7)
    tools = [build_tool_schema(explain=False)]
    kw = make_request_kwargs(cfg, prompt="Hello?", tools=tools)
    assert kw["model"] == "claude-opus-4-7"
    assert kw["temperature"] == 0.7
    assert kw["max_tokens"] == 1024
    assert kw["messages"] == [{"role": "user", "content": "Hello?"}]
    assert kw["tools"] == tools
    assert kw["tool_choice"] == {
        "type": "tool", "name": "submit_answer", "disable_parallel_tool_use": True,
    }
    assert "thinking" not in kw


def test_make_request_kwargs_with_thinking():
    cfg = _cfg(thinking="on", thinking_budget=4096, temperature=1.0, max_tokens=1024)
    tools = [build_tool_schema(explain=False)]
    kw = make_request_kwargs(cfg, prompt="Hello?", tools=tools)
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert kw["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}
    # max_tokens must exceed thinking_budget; auto-bump applies
    assert kw["max_tokens"] >= 4096 + 1024


def test_make_request_kwargs_auto_bumps_max_tokens():
    cfg = _cfg(thinking="on", thinking_budget=8000, max_tokens=1024)
    tools = [build_tool_schema(explain=False)]
    kw = make_request_kwargs(cfg, prompt="x", tools=tools)
    assert kw["max_tokens"] >= 8000 + 1024
