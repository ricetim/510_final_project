"""Tests for API call construction and execution (mocked)."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from run_openai import (
    Config, RateLimiter, build_output_schema, make_request_kwargs,
    parse_response, run_single,
)


def _no_pace() -> RateLimiter:
    return RateLimiter(rpm=100_000)


def _cfg(**overrides) -> Config:
    base = dict(
        input="in.csv", model="gpt-4o-mini", thinking="off",
        reasoning_effort="medium", explain=False, n=10, temperature=1.0,
        concurrency=5, limit=None, output=None, max_tokens=1024, rpm=400,
        question="",
    )
    base.update(overrides)
    return Config(**base)


def test_build_output_schema_explain_off():
    schema = build_output_schema(explain=False)
    props = schema["properties"]
    assert props["answer"]["enum"] == ["Yes", "No"]
    assert "explanation" not in props
    assert schema["required"] == ["answer"]
    assert schema["additionalProperties"] is False


def test_build_output_schema_explain_on():
    schema = build_output_schema(explain=True)
    props = schema["properties"]
    assert "explanation" in props
    assert set(schema["required"]) == {"answer", "explanation"}


def test_build_output_schema_is_independent_per_call():
    a = build_output_schema(explain=False)
    a["required"].append("explanation")
    b = build_output_schema(explain=False)
    assert b["required"] == ["answer"]


def test_make_request_kwargs_no_thinking():
    cfg = _cfg(thinking="off", model="gpt-4o-mini", temperature=0.7)
    schema = build_output_schema(explain=False)
    kw = make_request_kwargs(cfg, prompt="Hello?", output_schema=schema)
    assert kw["model"] == "gpt-4o-mini"
    assert kw["temperature"] == 0.7
    assert kw["input"] == "Hello?"
    assert kw["max_output_tokens"] == 1024
    assert kw["text"]["format"]["type"] == "json_schema"
    assert kw["text"]["format"]["schema"] == schema
    assert kw["text"]["format"]["strict"] is True
    assert "reasoning" not in kw


def test_make_request_kwargs_with_thinking_omits_temperature():
    """OpenAI reasoning models reject temperature — drop it from the request."""
    cfg = _cfg(thinking="on", reasoning_effort="medium", temperature=0.5)
    schema = build_output_schema(explain=False)
    kw = make_request_kwargs(cfg, prompt="Hello?", output_schema=schema)
    assert "temperature" not in kw
    assert kw["reasoning"] == {"effort": "medium"}


def test_make_request_kwargs_with_thinking_high_effort():
    cfg = _cfg(thinking="on", reasoning_effort="high")
    schema = build_output_schema(explain=False)
    kw = make_request_kwargs(cfg, prompt="x", output_schema=schema)
    assert kw["reasoning"] == {"effort": "high"}


def test_make_request_kwargs_auto_bumps_max_output_tokens_for_reasoning():
    """Reasoning runs need headroom above max_tokens for reasoning tokens."""
    cfg = _cfg(thinking="on", reasoning_effort="high", max_tokens=512)
    schema = build_output_schema(explain=False)
    kw = make_request_kwargs(cfg, prompt="x", output_schema=schema)
    assert kw["max_output_tokens"] >= 512 + 1024


def test_parse_response_success(responses_api_response):
    r = responses_api_response(answer="Yes", explanation="ok",
                                input_tokens=120, output_tokens=15,
                                cached_tokens=30)
    parsed = parse_response(r, latency_ms=234)
    assert parsed["answer"] == "Yes"
    assert parsed["explanation"] == "ok"
    assert parsed["stop_reason"] == "completed"
    assert parsed["input_tokens"] == 120
    assert parsed["output_tokens"] == 15
    assert parsed["cache_read_tokens"] == 30
    assert parsed["cache_creation_tokens"] == 0
    assert parsed["latency_ms"] == 234
    assert parsed["error"] == ""


def test_parse_response_incomplete(responses_api_incomplete):
    r = responses_api_incomplete(reason="max_output_tokens")
    parsed = parse_response(r, latency_ms=50)
    assert parsed["answer"] == ""
    assert parsed["explanation"] == ""
    assert "incomplete" in parsed["error"].lower()
    assert "max_output_tokens" in parsed["error"]


def test_parse_response_refusal(responses_api_refusal):
    r = responses_api_refusal(refusal_text="I cannot answer that.")
    parsed = parse_response(r, latency_ms=50)
    assert parsed["answer"] == ""
    assert "refusal" in parsed["error"].lower()
    assert "cannot answer" in parsed["error"]


def test_parse_response_invalid_json(responses_api_response):
    r = responses_api_response()
    r.output_text = "not valid json {"
    parsed = parse_response(r, latency_ms=10)
    assert parsed["answer"] == ""
    assert "json parse" in parsed["error"].lower()


def test_parse_response_missing_explanation(responses_api_response):
    """If the model omits explanation, store empty string, not None."""
    r = responses_api_response(answer="No", include_explanation=False)
    parsed = parse_response(r, latency_ms=10)
    assert parsed["answer"] == "No"
    assert parsed["explanation"] == ""


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
    client.responses.create = AsyncMock(return_value=response)
    return client


async def test_run_single_success(input_row, responses_api_response):
    client = _fake_client(responses_api_response(answer="Yes", explanation="ok"))
    cfg = _cfg()
    sem = asyncio.Semaphore(1)
    schema = build_output_schema(explain=False)
    row = await run_single(client, sem, _no_pace(), cfg, schema, input_row,
                            replicate_idx=3, run_id="abc-123")
    assert row["answer"] == "Yes"
    assert row["explanation"] == "ok"
    assert row["scenario_id"] == "00001"
    assert row["replicate_idx"] == 3
    assert row["model"] == cfg.model
    assert row["provider"] == "openai"
    assert row["thinking"] == "off"
    assert row["thinking_budget"] == ""
    assert row["reasoning_effort"] == ""  # thinking off → blank, not "medium"
    assert row["temperature"] == cfg.temperature
    assert row["explain_requested"] is False
    assert row["run_id"] == "abc-123"
    assert row["error"] == ""
    assert row["timestamp_utc"]


async def test_run_single_with_thinking_records_effort_blanks_temperature(
    input_row, responses_api_response,
):
    """When --thinking on: reasoning_effort populated, temperature blanked."""
    client = _fake_client(responses_api_response())
    cfg = _cfg(thinking="on", reasoning_effort="high", temperature=0.7)
    sem = asyncio.Semaphore(1)
    schema = build_output_schema(explain=False)
    row = await run_single(client, sem, _no_pace(), cfg, schema, input_row, 0, "rid")
    assert row["thinking"] == "on"
    assert row["reasoning_effort"] == "high"
    assert row["temperature"] == ""  # not sent, so not recorded


async def test_run_single_api_failure_captured(input_row):
    client = AsyncMock()
    client.responses.create = AsyncMock(side_effect=RuntimeError("boom"))
    cfg = _cfg()
    sem = asyncio.Semaphore(1)
    schema = build_output_schema(explain=False)
    row = await run_single(client, sem, _no_pace(), cfg, schema, input_row, 0, "rid")
    assert row["answer"] == ""
    assert "boom" in row["error"]
    assert row["stop_reason"] == ""
    assert row["scenario_id"] == "00001"
    assert row["replicate_idx"] == 0
    assert row["provider"] == "openai"


async def test_run_single_refusal_captured(input_row, responses_api_refusal):
    client = _fake_client(responses_api_refusal())
    cfg = _cfg()
    sem = asyncio.Semaphore(1)
    schema = build_output_schema(explain=False)
    row = await run_single(client, sem, _no_pace(), cfg, schema, input_row, 0, "rid")
    assert row["answer"] == ""
    assert "refusal" in row["error"].lower()


async def test_run_single_appends_question_to_prompt(input_row, responses_api_response):
    client = _fake_client(responses_api_response())
    question = " Was this acceptable behavior? Answer only with Yes or No."
    cfg = _cfg(question=question)
    sem = asyncio.Semaphore(1)
    schema = build_output_schema(explain=False)
    row = await run_single(client, sem, _no_pace(), cfg, schema, input_row, 0, "rid")
    assert row["question"] == question
    sent_kwargs = client.responses.create.await_args.kwargs
    assert sent_kwargs["input"] == input_row["variant_scenario"] + question
