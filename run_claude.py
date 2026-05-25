"""Run moral-variant prompts through the Anthropic Claude API."""
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
    rpm: int                # client-side requests-per-minute cap
    question: str           # appended to each variant_scenario before sending


class RateLimiter:
    """Pace request starts to at most `rpm` per 60 seconds.

    Enforces a minimum interval (60/rpm seconds) between successive
    `acquire()` returns. Multiple callers serialize through a lock but
    each one sleeps independently, so the limiter does not become a
    concurrency bottleneck — it just delays callers that arrive too
    early. Pairs with `asyncio.Semaphore` (which caps in-flight calls).
    """

    def __init__(self, rpm: int):
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")
        self._interval = 60.0 / rpm
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0  # monotonic time when the next acquire may proceed

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

    # Flag combo corrections + warnings.
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
    "model",
    "thinking",
    "thinking_budget",
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
    explain_label = "yes" if config.explain else "no"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = (
        f"{config.model}_think-{config.thinking}_n{config.n}"
        f"_explain-{explain_label}_{stamp}.csv"
    )
    # Anchor to the script's directory so output lands in the project's
    # results/ regardless of where the user invoked python from.
    return Path(__file__).resolve().parent / "results" / name


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
    p.add_argument("--rpm", type=int, default=45,
                   help="Client-side requests-per-minute cap (defaults to 45 to stay under "
                        "tier-1 50 RPM limits with headroom; raise if you're on a higher tier).")
    p.add_argument("--question", default="",
                   help="Text appended to each variant_scenario before sending to the model. "
                        "Use to test different framings without modifying the input CSV, e.g. "
                        '\' Was this acceptable behavior? Answer only with Yes or No.\'')
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


def build_tool_schema(explain: bool) -> dict:
    properties: dict = {
        "answer": {
            "type": "string",
            "enum": ["Yes", "No"],
            "description": "Your answer to the question.",
        },
    }
    if explain:
        properties["explanation"] = {
            "type": "string",
            "description": "Brief explanation of your reasoning.",
        }
    required = ["answer", "explanation"] if explain else ["answer"]
    return {
        "name": "submit_answer",
        "description": "Submit your answer to the moral scenario.",
        "input_schema": {
            "type": "object",
            "properties": properties,
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


THINKING_TOOL_INSTRUCTION = (
    "\n\nReturn your answer by calling the submit_answer tool. "
    "Do not respond with plain text."
)


def make_request_kwargs(config: Config, prompt: str, tools: list[dict]) -> dict:
    """Build kwargs for client.messages.create()."""
    thinking_on = config.thinking == "on"
    if thinking_on:
        max_tokens = max(config.max_tokens, config.thinking_budget + 1024)
        tool_choice = {"type": "auto", "disable_parallel_tool_use": True}
        user_content = prompt + THINKING_TOOL_INSTRUCTION
    else:
        max_tokens = config.max_tokens
        tool_choice = {
            "type": "tool",
            "name": "submit_answer",
            "disable_parallel_tool_use": True,
        }
        user_content = prompt
    kw: dict = {
        "model": config.model,
        "max_tokens": max_tokens,
        "temperature": config.temperature,
        "messages": [{"role": "user", "content": user_content}],
        "tools": tools,
        "tool_choice": tool_choice,
    }
    if thinking_on:
        kw["thinking"] = {
            "type": "enabled",
            "budget_tokens": config.thinking_budget,
        }
    return kw


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
    tools: list[dict],
    row: dict,
    replicate_idx: int,
    run_id: str,
) -> dict:
    """Issue one API call for one (row, replicate). Always returns a result row."""
    prompt = row["variant_scenario"] + config.question
    kwargs = make_request_kwargs(config, prompt, tools)
    metadata = {
        **{c: row.get(c, "") for c in RESULT_INPUT_COLUMNS},
        "replicate_idx": replicate_idx,
        "model": config.model,
        "thinking": config.thinking,
        "thinking_budget": config.thinking_budget if config.thinking == "on" else "",
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
        except Exception as exc:  # noqa: BLE001 — we want to capture all SDK errors
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
    load_dotenv()  # populate ANTHROPIC_API_KEY from .env if present
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

    tools = [build_tool_schema(explain=config.explain)]
    output_path = Path(config.output) if config.output else auto_output_path(config)
    run_id = str(uuid.uuid4())

    if config.thinking == "on":
        print(
            "note: extended thinking forces tool_choice=auto (API constraint); "
            "rare rows may return text instead of a tool call. Check the 'error' "
            "and 'answer' columns after the run.",
            file=sys.stderr,
        )

    n_calls = len(rows) * config.n
    est_minutes = n_calls / config.rpm
    print(
        f"run_id={run_id}  output={output_path}  "
        f"calls={n_calls} ({len(rows)} rows x {config.n} replicates)  "
        f"rpm={config.rpm} (~{est_minutes:.1f} min at rate-limit floor)",
        file=sys.stderr,
    )

    # Import the SDK lazily so unit tests don't require it to be importable.
    from anthropic import AsyncAnthropic

    # max_retries=10 (up from default 3) so the SDK can dig out of any 429
    # bursts that slip through proactive pacing — it uses the Retry-After
    # header from the API with exponential backoff.
    client = AsyncAnthropic(max_retries=10)
    semaphore = asyncio.Semaphore(config.concurrency)
    rate_limiter = RateLimiter(config.rpm)

    tasks = [
        asyncio.create_task(
            run_single(client, semaphore, rate_limiter, config,
                       tools, row, replicate_idx, run_id)
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
