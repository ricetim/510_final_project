"""Run moral-variant prompts through the Anthropic Claude API."""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    input: str
    model: str
    thinking: str           # "on" | "off"
    thinking_budget: int
    explain: bool
    n: int
    temperature: float
    concurrency: int
    limit: int | None
    output: str | None
    max_tokens: int


def parse_args(argv: list[str] | None = None) -> Config:
    p = argparse.ArgumentParser(
        description="Run moral-variant prompts through Claude.",
    )
    p.add_argument("--input", required=True, help="Path to input CSV.")
    p.add_argument("--model", required=True, help="Anthropic model ID, e.g. claude-opus-4-7.")
    p.add_argument("--thinking", choices=["on", "off"], default="off")
    p.add_argument("--thinking-budget", type=int, default=4096,
                   help="Tokens for extended thinking. Ignored when --thinking off.")
    p.add_argument("--explain", action="store_true",
                   help="Require an explanation alongside the answer.")
    p.add_argument("--n", type=int, default=10, help="Replicates per prompt.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N rows. For smoke tests.")
    p.add_argument("--output", default=None,
                   help="Output CSV path. Auto-generated under results/ if omitted.")
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="Messages API max_tokens. Auto-bumped above --thinking-budget when needed.")
    ns = p.parse_args(argv)
    return Config(
        input=ns.input,
        model=ns.model,
        thinking=ns.thinking,
        thinking_budget=ns.thinking_budget,
        explain=ns.explain,
        n=ns.n,
        temperature=ns.temperature,
        concurrency=ns.concurrency,
        limit=ns.limit,
        output=ns.output,
        max_tokens=ns.max_tokens,
    )


def build_tool_schema(explain: bool) -> dict:
    required = ["answer", "explanation"] if explain else ["answer"]
    return {
        "name": "submit_answer",
        "description": "Submit your answer to the moral scenario.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "enum": ["Yes", "No"],
                    "description": "Your answer to the question.",
                },
                "explanation": {
                    "type": "string",
                    "description": "Brief explanation of your reasoning.",
                },
            },
            "required": required,
        },
    }


def parse_response(response, latency_ms: int) -> dict:
    """Extract answer/explanation/usage from an Anthropic Messages response."""
    tool_block = next(
        (b for b in response.content if getattr(b, "type", None) == "tool_use"
         and getattr(b, "name", None) == "submit_answer"),
        None,
    )
    usage = response.usage
    base = {
        "stop_reason": response.stop_reason or "",
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "latency_ms": latency_ms,
    }
    if tool_block is None:
        return {
            **base,
            "answer": "",
            "explanation": "",
            "error": "no tool_use block in response",
        }
    payload = tool_block.input or {}
    return {
        **base,
        "answer": payload.get("answer", "") or "",
        "explanation": payload.get("explanation", "") or "",
        "error": "",
    }
