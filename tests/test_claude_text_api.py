"""Tests for API call construction and execution (mocked) for run_claude_text.py."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from run_claude_text import (
    Config, RateLimiter, make_request_kwargs, parse_response, run_single,
)


def _no_pace() -> RateLimiter:
    return RateLimiter(rpm=100_000)


def _cfg(**overrides) -> Config:
    base = dict(
        input="in.csv", model="claude-haiku-4-5-20251001", thinking="off",
        thinking_budget=4096, explain=False, n=10, temperature=1.0,
        concurrency=5, limit=None, output=None, max_tokens=1024, rpm=45,
        question="",
    )
    base.update(overrides)
    return Config(**base)


def test_make_request_kwargs_no_thinking_no_tools():
    cfg = _cfg(thinking="off", temperature=0.7)
    kw = make_request_kwargs(cfg, prompt="Hello?")
    assert kw["model"] == "claude-haiku-4-5-20251001"
    assert kw["temperature"] == 0.7
    assert kw["max_tokens"] == 1024
    assert kw["messages"] == [{"role": "user", "content": "Hello?"}]
    assert "tools" not in kw
    assert "tool_choice" not in kw
    assert "thinking" not in kw


def test_make_request_kwargs_with_thinking_omits_tools():
    cfg = _cfg(thinking="on", thinking_budget=4096, temperature=1.0)
    kw = make_request_kwargs(cfg, prompt="Hello?")
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert "tools" not in kw
    assert "tool_choice" not in kw
    assert kw["max_tokens"] >= 4096 + 1024


def test_make_request_kwargs_auto_bumps_max_tokens():
    cfg = _cfg(thinking="on", thinking_budget=8000, max_tokens=1024)
    kw = make_request_kwargs(cfg, prompt="x")
    assert kw["max_tokens"] >= 8000 + 1024


def test_parse_response_yes(text_only_response):
    r = text_only_response(text="Yes")
    parsed = parse_response(r, latency_ms=234)
    assert parsed["answer"] == "Yes"
    assert parsed["explanation"] == ""
    assert parsed["error"] == ""
    assert parsed["latency_ms"] == 234


def test_parse_response_no(text_only_response):
    r = text_only_response(text="No")
    parsed = parse_response(r, latency_ms=10)
    assert parsed["answer"] == "No"
    assert parsed["error"] == ""


def test_parse_response_yes_with_period(text_only_response):
    r = text_only_response(text="Yes.")
    parsed = parse_response(r, latency_ms=10)
    assert parsed["answer"] == "Yes"
    assert parsed["error"] == ""


def test_parse_response_case_insensitive(text_only_response):
    r = text_only_response(text="yes")
    parsed = parse_response(r, latency_ms=10)
    assert parsed["answer"] == "Yes"
    assert parsed["error"] == ""


def test_parse_response_extra_whitespace(text_only_response):
    r = text_only_response(text="  Yes  ")
    parsed = parse_response(r, latency_ms=10)
    assert parsed["answer"] == "Yes"
    assert parsed["error"] == ""


def test_parse_response_unexpected_output(text_only_response):
    r = text_only_response(text="Yes, because the action was justified.")
    parsed = parse_response(r, latency_ms=10)
    assert parsed["answer"] == ""
    assert "unexpected output" in parsed["error"].lower()
    assert "Yes, because" in parsed["error"]


def test_parse_response_no_text_block(tool_use_response):
    """A response with only a tool_use block (no text) yields an error."""
    r = tool_use_response()  # has content=[tool_use_block] only
    parsed = parse_response(r, latency_ms=10)
    assert parsed["answer"] == ""
    assert "no text block" in parsed["error"].lower()


@pytest.fixture
def input_row() -> dict:
    return {
        "scenario_id": "00001",
        "original_scenario": "orig",
        "race_variant": "white",
        "income_variant": "low",
        "variant_scenario": "Would you do X? Answer only with Yes or No.",
    }


def _fake_client(response):
    client = AsyncMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


async def test_run_single_success(input_row, text_only_response):
    client = _fake_client(text_only_response(text="Yes"))
    cfg = _cfg()
    sem = asyncio.Semaphore(1)
    row = await run_single(client, sem, _no_pace(), cfg, input_row,
                            replicate_idx=3, run_id="abc-123")
    assert row["answer"] == "Yes"
    assert row["explanation"] == ""
    assert row["scenario_id"] == "00001"
    assert row["replicate_idx"] == 3
    assert row["model"] == cfg.model
    assert row["provider"] == "anthropic"
    assert row["thinking"] == "off"
    assert row["thinking_budget"] == ""
    assert row["reasoning_effort"] == ""
    assert row["temperature"] == cfg.temperature
    assert row["error"] == ""
    sent_kwargs = client.messages.create.await_args.kwargs
    assert "tools" not in sent_kwargs
    assert "tool_choice" not in sent_kwargs


async def test_run_single_with_thinking_records_budget(input_row, text_only_response):
    client = _fake_client(text_only_response(text="No"))
    cfg = _cfg(thinking="on", thinking_budget=2048)
    sem = asyncio.Semaphore(1)
    row = await run_single(client, sem, _no_pace(), cfg, input_row, 0, "rid")
    assert row["thinking"] == "on"
    assert row["thinking_budget"] == 2048
    assert row["answer"] == "No"


async def test_run_single_appends_question_to_prompt(input_row, text_only_response):
    client = _fake_client(text_only_response(text="Yes"))
    question = " Was this acceptable behavior? Answer only with Yes or No."
    cfg = _cfg(question=question)
    sem = asyncio.Semaphore(1)
    row = await run_single(client, sem, _no_pace(), cfg, input_row, 0, "rid")
    assert row["question"] == question
    sent_kwargs = client.messages.create.await_args.kwargs
    assert sent_kwargs["messages"][0]["content"] == input_row["variant_scenario"] + question


async def test_run_single_api_failure_captured(input_row):
    client = AsyncMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("boom"))
    cfg = _cfg()
    sem = asyncio.Semaphore(1)
    row = await run_single(client, sem, _no_pace(), cfg, input_row, 0, "rid")
    assert row["answer"] == ""
    assert "boom" in row["error"]
    assert row["provider"] == "anthropic"


async def test_run_single_unexpected_output_captured(input_row, text_only_response):
    client = _fake_client(text_only_response(text="I cannot answer that question."))
    cfg = _cfg()
    sem = asyncio.Semaphore(1)
    row = await run_single(client, sem, _no_pace(), cfg, input_row, 0, "rid")
    assert row["answer"] == ""
    assert "unexpected output" in row["error"].lower()
