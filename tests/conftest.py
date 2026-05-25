"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def tmp_input_csv(tmp_path: Path) -> Path:
    """Create a 3-row input CSV with the canonical schema."""
    p = tmp_path / "input.csv"
    p.write_text(
        "scenario_id,original_scenario,race_variant,income_variant,variant_scenario\n"
        '00001,"orig A",white,low,"prompt text 1. Answer only with Yes or No."\n'
        '00001,"orig A",black,low,"prompt text 2. Answer only with Yes or No."\n'
        '00002,"orig B",latino,high,"prompt text 3. Answer only with Yes or No."\n'
    )
    return p


def _usage(input_tokens=100, output_tokens=20, cache_read=0, cache_create=0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_create,
    )


def make_tool_use_response(answer="Yes", explanation="because reasons",
                            stop_reason="tool_use", **usage_kwargs):
    """Build a mock Anthropic response with a tool_use content block."""
    tool_block = SimpleNamespace(
        type="tool_use",
        name="submit_answer",
        input={"answer": answer, "explanation": explanation},
    )
    return SimpleNamespace(
        content=[tool_block],
        stop_reason=stop_reason,
        usage=_usage(**usage_kwargs),
    )


def make_text_only_response(text="No tool call made.", stop_reason="end_turn"):
    text_block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(
        content=[text_block],
        stop_reason=stop_reason,
        usage=_usage(),
    )


@pytest.fixture
def tool_use_response():
    return make_tool_use_response


@pytest.fixture
def text_only_response():
    return make_text_only_response


def _openai_usage(input_tokens=100, output_tokens=20, cached_tokens=0,
                   reasoning_tokens=0):
    """Build a SimpleNamespace mimicking response.usage from the Responses API."""
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
    )


def make_responses_api_response(answer="Yes", explanation="because reasons",
                                  status="completed", include_explanation=True,
                                  **usage_kwargs):
    """Build a mock OpenAI Responses API response with structured-output JSON.

    output_text holds the JSON string the SDK would assemble from the
    response's text format. We don't model the full output items array —
    parse_response only reads output_text, status, and usage.
    """
    import json
    payload = {"answer": answer}
    if include_explanation:
        payload["explanation"] = explanation
    return SimpleNamespace(
        output_text=json.dumps(payload),
        status=status,
        usage=_openai_usage(**usage_kwargs),
        # Used by refusal/error-path code; default to empty
        output=[],
    )


def make_responses_api_incomplete(reason="max_output_tokens"):
    """Build a mock response where status != 'completed'."""
    return SimpleNamespace(
        output_text="",
        status="incomplete",
        incomplete_details=SimpleNamespace(reason=reason),
        usage=_openai_usage(),
        output=[],
    )


def make_responses_api_refusal(refusal_text="I cannot answer that."):
    """Build a mock response where the model issued a refusal item."""
    refusal_item = SimpleNamespace(type="refusal", refusal=refusal_text)
    return SimpleNamespace(
        output_text="",
        status="completed",
        usage=_openai_usage(),
        output=[refusal_item],
    )


@pytest.fixture
def responses_api_response():
    return make_responses_api_response


@pytest.fixture
def responses_api_incomplete():
    return make_responses_api_incomplete


@pytest.fixture
def responses_api_refusal():
    return make_responses_api_refusal
