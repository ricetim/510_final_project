"""Run moral-variant prompts through the OpenAI Responses API."""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


PROVIDER = "openai"


@dataclass(frozen=True)
class Config:
    input: str
    model: str
    thinking: str               # "on" | "off"
    reasoning_effort: str       # "low" | "medium" | "high"; only used when thinking == "on"
    explain: bool
    n: int
    temperature: float
    concurrency: int
    limit: int | None
    output: str | None
    max_tokens: int
    rpm: int                    # client-side requests-per-minute cap
    question: str               # appended to each variant_scenario before sending


class RateLimiter:
    """Pace request starts to at most `rpm` per 60 seconds.

    Same semantics as RateLimiter in run_claude.py. Duplicated here per the
    OpenAI runner design — `study_io.py` extraction is a follow-up.
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


# OpenAI reasoning models (o-series, gpt-5*) accept `reasoning.effort`;
# chat models (gpt-4o*, gpt-4-turbo, gpt-3.5-*, ...) 400 on it. Naming pattern
# is stable enough to gate on client-side — the SDK does not pre-validate.
REASONING_MODEL_RE = re.compile(r"^(o\d|gpt-5)")


def preflight(config: Config) -> Config:
    """Validate env + flag combinations. Returns a possibly-corrected Config."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise ConfigError(
            "OPENAI_API_KEY is not set. Add it to .env or export it."
        )
    if not Path(config.input).exists():
        raise ConfigError(f"input CSV not found: {config.input}")

    if config.thinking == "on" and not REASONING_MODEL_RE.match(config.model):
        raise ConfigError(
            f"--thinking on requires a reasoning model (o-series or gpt-5*); "
            f"{config.model!r} is a chat model and the Responses API will "
            f"reject 'reasoning.effort' for it. Either drop --thinking on or "
            f"pick a reasoning model (e.g. o4-mini, gpt-5-mini)."
        )

    if config.thinking == "off" and config.reasoning_effort != "medium":
        # medium is the default; only warn if user set it explicitly while
        # also turning thinking off.
        print(
            "warning: --reasoning-effort is ignored when --thinking off",
            file=sys.stderr,
        )
    return config


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
    """Read input CSV, validate schema, return rows as dicts.

    Duplicates run_claude.load_input_rows verbatim per the spec.
    """
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
    reasoning_label = config.reasoning_effort if config.thinking == "on" else "off"
    name = (
        f"{config.model}_reasoning-{reasoning_label}_n{config.n}"
        f"_explain-{explain_label}_{stamp}.csv"
    )
    return Path(__file__).resolve().parent / "results" / name


def parse_args(argv: list[str] | None = None) -> Config:
    p = argparse.ArgumentParser(
        description="Run moral-variant prompts through OpenAI's Responses API.",
    )
    p.add_argument("--input", required=True, help="Path to input CSV.")
    p.add_argument("--model", required=True, help="OpenAI model ID, e.g. gpt-4o-mini, o4-mini.")
    p.add_argument("--thinking", choices=["on", "off"], default="off")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                   default="medium",
                   help="Reasoning effort for o-series / GPT-5 models. "
                        "Ignored when --thinking off.")
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
                   help="Responses API max_output_tokens. Auto-bumped for reasoning runs.")
    p.add_argument("--rpm", type=int, default=400,
                   help="Client-side requests-per-minute cap (defaults to 400; "
                        "raise/lower based on your OpenAI tier).")
    p.add_argument("--question", default="",
                   help="Text appended to each variant_scenario before sending to the model. "
                        "Use to test different framings without modifying the input CSV.")
    ns = p.parse_args(argv)
    return Config(
        input=ns.input,
        model=ns.model,
        thinking=ns.thinking,
        reasoning_effort=ns.reasoning_effort,
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


class ResultWriter:
    """Writes the header on construction; one row per call to write().

    Duplicates run_claude.ResultWriter verbatim per the spec.
    """

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


def build_output_schema(explain: bool) -> dict:
    """JSON Schema for forced structured output via Responses API.

    Mirrors the spirit of build_tool_schema in run_claude.py: the
    explanation key is omitted from properties entirely when explain=False
    so the model cannot volunteer one and contaminate answer-only runs.
    """
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
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def make_request_kwargs(config: Config, prompt: str, output_schema: dict) -> dict:
    """Build kwargs for client.responses.create()."""
    thinking_on = config.thinking == "on"
    # Reasoning runs need headroom for reasoning tokens above the user-requested
    # output budget. 1024 is a conservative floor; the user can raise --max-tokens.
    if thinking_on:
        max_output_tokens = config.max_tokens + 1024
    else:
        max_output_tokens = config.max_tokens
    kw: dict = {
        "model": config.model,
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "moral_answer",
                "schema": output_schema,
                "strict": True,
            },
        },
        "max_output_tokens": max_output_tokens,
    }
    if thinking_on:
        # Reasoning models reject `temperature`; omit it entirely.
        kw["reasoning"] = {"effort": config.reasoning_effort}
    else:
        kw["temperature"] = config.temperature
    return kw


def parse_response(response, latency_ms: int) -> dict:
    """Extract answer/explanation/usage from an OpenAI Responses API response."""
    usage = response.usage
    base = {
        "stop_reason": getattr(response, "status", "") or "",
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_read_tokens": getattr(
            getattr(usage, "input_tokens_details", None), "cached_tokens", 0
        ) or 0,
        "cache_creation_tokens": 0,  # OpenAI cache is automatic; no creation cost
        "latency_ms": latency_ms,
    }

    # Check for incomplete response (e.g., max_output_tokens hit)
    if getattr(response, "status", "") == "incomplete":
        reason = getattr(
            getattr(response, "incomplete_details", None), "reason", "unknown"
        )
        return {
            **base,
            "answer": "",
            "explanation": "",
            "error": f"incomplete: {reason}",
        }

    # Check for refusal items in output
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "refusal":
            return {
                **base,
                "answer": "",
                "explanation": "",
                "error": f"model refusal: {getattr(item, 'refusal', '')}",
            }

    # Parse the structured JSON output
    output_text = getattr(response, "output_text", "") or ""
    try:
        payload = json.loads(output_text)
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            **base,
            "answer": "",
            "explanation": "",
            "error": f"json parse error: {exc}",
        }
    return {
        **base,
        "answer": payload.get("answer", "") or "",
        "explanation": payload.get("explanation", "") or "",
        "error": "",
    }


async def run_single(
    client,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    config: Config,
    output_schema: dict,
    row: dict,
    replicate_idx: int,
    run_id: str,
) -> dict:
    """Issue one API call for one (row, replicate). Always returns a result row."""
    prompt = row["variant_scenario"] + config.question
    kwargs = make_request_kwargs(config, prompt, output_schema)
    metadata = {
        **{c: row.get(c, "") for c in RESULT_INPUT_COLUMNS},
        "replicate_idx": replicate_idx,
        "provider": PROVIDER,
        "model": config.model,
        "thinking": config.thinking,
        "thinking_budget": "",  # n/a for OpenAI
        "reasoning_effort": config.reasoning_effort if config.thinking == "on" else "",
        "temperature": config.temperature if config.thinking == "off" else "",
        "explain_requested": config.explain,
        "question": config.question,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_id,
    }
    async with semaphore:
        await rate_limiter.acquire()
        t0 = time.monotonic()
        try:
            response = await client.responses.create(**kwargs)
            latency_ms = int((time.monotonic() - t0) * 1000)
        except Exception as exc:  # noqa: BLE001 — capture all SDK errors
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

    output_schema = build_output_schema(explain=config.explain)
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

    from openai import AsyncOpenAI
    from tqdm.asyncio import tqdm_asyncio

    client = AsyncOpenAI(max_retries=10)
    semaphore = asyncio.Semaphore(config.concurrency)
    rate_limiter = RateLimiter(config.rpm)

    tasks = [
        asyncio.create_task(
            run_single(client, semaphore, rate_limiter, config,
                       output_schema, row, replicate_idx, run_id)
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
