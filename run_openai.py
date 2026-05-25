"""Run moral-variant prompts through the OpenAI Responses API."""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
import time
from dataclasses import dataclass, replace
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


def preflight(config: Config) -> Config:
    """Validate env + flag combinations. Returns a possibly-corrected Config."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise ConfigError(
            "OPENAI_API_KEY is not set. Add it to .env or export it."
        )
    if not Path(config.input).exists():
        raise ConfigError(f"input CSV not found: {config.input}")

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


# Placeholder for the API call functions — added in Chunk 5.
# Placeholder for run_single and run_study — added in Chunk 5.
# Placeholder for main — added in Chunk 5.
