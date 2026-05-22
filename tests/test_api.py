"""Tests for API call construction and execution (mocked)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from run_claude import (
    Config, RateLimiter, build_tool_schema, make_request_kwargs, run_single,
)


def _no_pace() -> RateLimiter:
    """RateLimiter that never sleeps — for tests that don't care about pacing."""
    return RateLimiter(rpm=100_000)


def _cfg(**overrides) -> Config:
    base = dict(
        input="in.csv", model="claude-opus-4-7", thinking="off",
        thinking_budget=4096, explain=False, n=10, temperature=1.0,
        concurrency=5, limit=None, output=None, max_tokens=1024, rpm=45,
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
    # Anthropic API rejects forced tool_choice (type "tool" or "any") when
    # thinking is enabled. Must use "auto" and rely on prompt instruction.
    assert kw["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    # max_tokens must exceed thinking_budget; auto-bump applies
    assert kw["max_tokens"] >= 4096 + 1024


def test_make_request_kwargs_with_thinking_appends_tool_instruction():
    """When thinking is on, the prompt must include an explicit instruction
    to call the submit_answer tool — otherwise the model may return plain text."""
    cfg = _cfg(thinking="on")
    tools = [build_tool_schema(explain=False)]
    kw = make_request_kwargs(cfg, prompt="Original prompt.", tools=tools)
    content = kw["messages"][0]["content"]
    assert content.startswith("Original prompt.")
    assert "submit_answer" in content


def test_make_request_kwargs_auto_bumps_max_tokens():
    cfg = _cfg(thinking="on", thinking_budget=8000, max_tokens=1024)
    tools = [build_tool_schema(explain=False)]
    kw = make_request_kwargs(cfg, prompt="x", tools=tools)
    assert kw["max_tokens"] >= 8000 + 1024


@pytest.fixture
def input_row() -> dict:
    return {
        "base_id": "PRU-001",
        "original_scenario_a": "orig",
        "variant_description": "variant 1",
        "full_binary_prompt": "Would you do X? Answer only with Yes or No.",
        "virtue": "Prudence",
    }


def _fake_client(response):
    client = AsyncMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


async def test_run_single_success(input_row, tool_use_response):
    client = _fake_client(tool_use_response(answer="Yes", explanation="ok"))
    cfg = _cfg()
    sem = asyncio.Semaphore(1)
    tools = [build_tool_schema(explain=False)]
    row = await run_single(client, sem, _no_pace(), cfg, tools, input_row,
                            replicate_idx=3, run_id="abc-123")
    assert row["answer"] == "Yes"
    assert row["explanation"] == "ok"
    assert row["base_id"] == "PRU-001"
    assert row["replicate_idx"] == 3
    assert row["model"] == cfg.model
    assert row["thinking"] == "off"
    assert row["thinking_budget"] == ""           # blanked when thinking=off
    assert row["temperature"] == cfg.temperature
    assert row["explain_requested"] is False
    assert row["run_id"] == "abc-123"
    assert row["error"] == ""
    assert row["timestamp_utc"]                    # non-empty


async def test_run_single_with_thinking_records_budget(input_row, tool_use_response):
    client = _fake_client(tool_use_response())
    cfg = _cfg(thinking="on", thinking_budget=2048)
    sem = asyncio.Semaphore(1)
    tools = [build_tool_schema(explain=False)]
    row = await run_single(client, sem, _no_pace(), cfg, tools, input_row, 0, "rid")
    assert row["thinking"] == "on"
    assert row["thinking_budget"] == 2048


async def test_run_single_api_failure_captured(input_row):
    client = AsyncMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("boom"))
    cfg = _cfg()
    sem = asyncio.Semaphore(1)
    tools = [build_tool_schema(explain=False)]
    row = await run_single(client, sem, _no_pace(), cfg, tools, input_row, 0, "rid")
    assert row["answer"] == ""
    assert "boom" in row["error"]
    assert row["stop_reason"] == ""
    # Row still has all metadata columns:
    assert row["base_id"] == "PRU-001"
    assert row["replicate_idx"] == 0


async def test_run_single_no_tool_use_block(input_row, text_only_response):
    client = _fake_client(text_only_response())
    cfg = _cfg()
    sem = asyncio.Semaphore(1)
    tools = [build_tool_schema(explain=False)]
    row = await run_single(client, sem, _no_pace(), cfg, tools, input_row, 0, "rid")
    assert row["answer"] == ""
    assert "no tool_use block" in row["error"].lower()
    assert row["stop_reason"] == "end_turn"


async def test_rate_limiter_paces_calls():
    """5 calls at 600 rpm (100ms interval) should take at least 400ms total."""
    limiter = RateLimiter(rpm=600)
    t0 = asyncio.get_event_loop().time()
    await asyncio.gather(*[limiter.acquire() for _ in range(5)])
    elapsed = asyncio.get_event_loop().time() - t0
    # First acquire is immediate; remaining 4 must wait one interval each.
    assert elapsed >= 0.4, f"expected >= 0.4s, got {elapsed:.3f}s"
    # And not absurdly slow either (loose upper bound to avoid flake):
    assert elapsed < 1.0, f"limiter is too slow: {elapsed:.3f}s"


def test_rate_limiter_rejects_zero_rpm():
    with pytest.raises(ValueError):
        RateLimiter(rpm=0)
