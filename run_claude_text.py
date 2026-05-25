"""Run moral-variant prompts through the Anthropic Claude API using raw text output.

This is a cost-optimized variant of run_claude.py that omits the tool-use
mechanism: requests are sent without tools/tool_choice, and responses are
parsed by stripping whitespace from the first text block and validating
against {"Yes", "No"}. The result is ~85% lower input token cost (no hidden
tool-use system prompt) and ~80% lower output token cost (just the answer,
no tool_use wrapper), at the cost of occasional model outputs that don't
match the expected format (recorded in the error column).

See CLAUDE.md for the cost/quality tradeoff and when to choose each.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio


PROVIDER = "anthropic"


@dataclass(frozen=True)
class Config:
    input: str
    model: str
    thinking: str
    thinking_budget: int
    explain: bool
    n: int
    temperature: float
    concurrency: int
    limit: int | None
    output: str | None
    max_tokens: int
    rpm: int
    question: str


class RateLimiter:
    """Pace request starts to at most `rpm` per 60 seconds.

    Duplicated from run_claude.py per the project's "copy-and-adapt"
    philosophy — `study_io.py` extraction is a still-deferred follow-up.
    """

    def __init__(self, rpm: int):
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")
        self._interval = 60.0 / rpm
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self._interval
        if wait > 0:
            await asyncio.sleep(wait)


class ConfigError(RuntimeError):
    pass


def preflight(config: Config) -> Config:
    """Validate env + flag combinations. Returns a possibly-corrected Config."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ConfigError(
            "ANTHROPIC_API_KEY is not set. Add it to .env or export it."
        )
    if not Path(config.input).exists():
        raise ConfigError(f"input CSV not found: {config.input}")
    if config.explain:
        raise ConfigError(
            "text mode does not support --explain; use run_claude.py if you "
            "need explanations alongside answers."
        )

    corrected = config
    if config.thinking == "on" and config.temperature != 1.0:
        print(
            f"warning: extended thinking requires temperature=1.0; "
            f"overriding --temperature {config.temperature} -> 1.0",
            file=sys.stderr,
        )
        corrected = replace(corrected, temperature=1.0)
    if config.thinking == "off" and config.thinking_budget != 4096:
        print(
            "warning: --thinking-budget is ignored when --thinking off",
            file=sys.stderr,
        )
    return corrected


RESULT_INPUT_COLUMNS: list[str] = [
    "scenario_id",
    "original_scenario",
    "race_variant",
    "income_variant",
    "variant_scenario",
]


OUTPUT_COLUMNS: list[str] = [
    *RESULT_INPUT_COLUMNS,
    "replicate_idx",
    "provider",
    "model",
    "thinking",
    "thinking_budget",
    "reasoning_effort",
    "temperature",
    "explain_requested",
    "question",
    "answer",
    "explanation",
    "stop_reason",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "latency_ms",
    "timestamp_utc",
    "error",
    "run_id",
]


def load_input_rows(path: str, limit: int | None) -> list[dict]:
    """Read input CSV, validate schema, return rows as dicts."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in RESULT_INPUT_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Input CSV missing required column(s): {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("Input CSV has no data rows.")
    if limit is not None:
        rows = rows[:limit]
    return rows


def auto_output_path(config: Config) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    # explain is rejected in preflight, so always "no" in the filename.
    name = (
        f"{config.model}_text_think-{config.thinking}_n{config.n}"
        f"_explain-no_{stamp}.csv"
    )
    return Path(__file__).resolve().parent / "results" / name


def parse_args(argv: list[str] | None = None) -> Config:
    p = argparse.ArgumentParser(
        description="Run moral-variant prompts through Claude using raw text output (no tool use).",
    )
    p.add_argument("--input", required=True, help="Path to input CSV.")
    p.add_argument("--model", required=True, help="Anthropic model ID, e.g. claude-haiku-4-5-20251001.")
    p.add_argument("--thinking", choices=["on", "off"], default="off")
    p.add_argument("--thinking-budget", type=int, default=4096,
                   help="Tokens for extended thinking. Ignored when --thinking off.")
    p.add_argument("--explain", action="store_true",
                   help="REJECTED in text mode; use run_claude.py for explanations.")
    p.add_argument("--n", type=int, default=10, help="Replicates per prompt.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N rows. For smoke tests.")
    p.add_argument("--output", default=None,
                   help="Output CSV path. Auto-generated under results/ if omitted.")
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="Messages API max_tokens. Auto-bumped above --thinking-budget when needed.")
    p.add_argument("--rpm", type=int, default=45,
                   help="Client-side requests-per-minute cap.")
    p.add_argument("--question", default="",
                   help="Text appended to each variant_scenario before sending to the model.")
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
        rpm=ns.rpm,
        question=ns.question,
    )


def make_request_kwargs(config: Config, prompt: str) -> dict:
    """Build kwargs for client.messages.create() — no tools, raw text response."""
    thinking_on = config.thinking == "on"
    if thinking_on:
        max_tokens = max(config.max_tokens, config.thinking_budget + 1024)
    else:
        max_tokens = config.max_tokens
    kw: dict = {
        "model": config.model,
        "max_tokens": max_tokens,
        "temperature": config.temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if thinking_on:
        kw["thinking"] = {"type": "enabled", "budget_tokens": config.thinking_budget}
    return kw


def parse_response(response, latency_ms: int) -> dict:
    """Extract Yes/No from the first text content block; record error otherwise."""
    text_block = next(
        (b for b in response.content if getattr(b, "type", None) == "text"),
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
    if text_block is None:
        return {
            **base,
            "answer": "",
            "explanation": "",
            "error": "no text block in response",
        }
    raw = (text_block.text or "").strip()
    normalized = raw.rstrip(".!?").strip()
    if normalized.lower() == "yes":
        return {**base, "answer": "Yes", "explanation": "", "error": ""}
    if normalized.lower() == "no":
        return {**base, "answer": "No", "explanation": "", "error": ""}
    return {
        **base,
        "answer": "",
        "explanation": "",
        "error": f"unexpected output: {raw[:100]}",
    }


class ResultWriter:
    """Writes the header on construction; one row per call to write()."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._f,
            fieldnames=OUTPUT_COLUMNS,
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        self._writer.writeheader()
        self._f.flush()

    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


async def run_single(
    client,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    config: Config,
    row: dict,
    replicate_idx: int,
    run_id: str,
) -> dict:
    """Issue one API call for one (row, replicate). Always returns a result row."""
    prompt = row["variant_scenario"] + config.question
    kwargs = make_request_kwargs(config, prompt)
    metadata = {
        **{c: row.get(c, "") for c in RESULT_INPUT_COLUMNS},
        "replicate_idx": replicate_idx,
        "provider": PROVIDER,
        "model": config.model,
        "thinking": config.thinking,
        "thinking_budget": config.thinking_budget if config.thinking == "on" else "",
        "reasoning_effort": "",
        "temperature": config.temperature,
        "explain_requested": config.explain,
        "question": config.question,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_id,
    }
    async with semaphore:
        await rate_limiter.acquire()
        t0 = time.monotonic()
        try:
            response = await client.messages.create(**kwargs)
            latency_ms = int((time.monotonic() - t0) * 1000)
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.monotonic() - t0) * 1000)
            return {
                **metadata,
                "answer": "",
                "explanation": "",
                "stop_reason": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "latency_ms": latency_ms,
                "error": f"{type(exc).__name__}: {exc}",
            }
    parsed = parse_response(response, latency_ms)
    return {**metadata, **parsed}


async def run_study(config: Config) -> int:
    """Top-level orchestrator. Returns process exit code."""
    load_dotenv()
    try:
        config = preflight(config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        rows = load_input_rows(config.input, config.limit)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_path = Path(config.output) if config.output else auto_output_path(config)
    run_id = str(uuid.uuid4())

    n_calls = len(rows) * config.n
    est_minutes = n_calls / config.rpm
    print(
        f"run_id={run_id}  output={output_path}  "
        f"calls={n_calls} ({len(rows)} rows x {config.n} replicates)  "
        f"rpm={config.rpm} (~{est_minutes:.1f} min at rate-limit floor)",
        file=sys.stderr,
    )

    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(max_retries=10)
    semaphore = asyncio.Semaphore(config.concurrency)
    rate_limiter = RateLimiter(config.rpm)

    tasks = [
        asyncio.create_task(
            run_single(client, semaphore, rate_limiter, config,
                       row, replicate_idx, run_id)
        )
        for row in rows
        for replicate_idx in range(config.n)
    ]

    with ResultWriter(output_path) as writer:
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            result = await coro
            writer.write(result)

    print(f"done: wrote {len(tasks)} rows to {output_path}", file=sys.stderr)
    return 0


def main() -> None:
    config = parse_args()
    exit_code = asyncio.run(run_study(config))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
