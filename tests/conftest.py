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
        "base_id,original_scenario_a,variant_description,full_binary_prompt,virtue\n"
        'PRU-001,"orig A","variant 1","prompt text 1. Answer only with Yes or No.",Prudence\n'
        'PRU-001,"orig A","variant 2","prompt text 2. Answer only with Yes or No.",Prudence\n'
        'JUS-001,"orig B","variant 3","prompt text 3. Answer only with Yes or No.",Justice\n'
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
