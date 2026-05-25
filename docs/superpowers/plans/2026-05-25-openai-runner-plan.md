# OpenAI Runner Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `run_openai.py` as a parallel-sibling to `run_claude.py` so the same moral-variant dataset can be run through OpenAI's Responses API, producing CSVs with a unified schema for cross-provider comparison.

**Architecture:** Hybrid mirror — same top-level function shape as `run_claude.py` (`parse_args`, `preflight`, `load_input_rows`, `make_request_kwargs`, `parse_response`, `run_single`, `run_study`), idiomatic OpenAI internals (Responses API with `text.format=json_schema`, `reasoning={"effort": ...}`). Shared utilities (`ResultWriter`, `RateLimiter`, `load_input_rows`, `OUTPUT_COLUMNS`) are duplicated for now per the spec — extraction to `study_io.py` is a follow-up once divergence (or lack thereof) is empirically visible.

**Tech Stack:** Python 3.10+, `openai>=1.50.0` (Responses API), `asyncio`, `pytest` with `pytest-asyncio`, `python-dotenv`, `tqdm`.

**Spec:** `docs/superpowers/specs/2026-05-25-openai-runner-design.md`

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `run_claude.py` | Modify | Backport: add `provider` + `reasoning_effort` columns, `PROVIDER` constant |
| `run_openai.py` | Create | OpenAI port — hybrid-mirror sibling of `run_claude.py` |
| `pyproject.toml` | Modify | Add `openai>=1.50.0` dependency |
| `tests/conftest.py` | Modify | Add `make_responses_api_response()` mock helper |
| `tests/test_openai_pure.py` | Create | parse_args, preflight, schema, auto_output_path, output column invariants |
| `tests/test_openai_api.py` | Create | make_request_kwargs, parse_response, run_single (all mocked) |
| `tests/test_pure.py` | Modify | Update column count assertions for new 25-column schema |
| `tests/test_api.py` | Modify | Add assertions that `provider="anthropic"` and `reasoning_effort=""` are recorded |
| `CLAUDE.md` | Modify | Document `--reasoning-effort` flag, update column count, note dual-runner architecture |
| `.env` | User-managed | User adds `OPENAI_API_KEY` (no script change) |

---

## Chunk 1: Backport schema changes to `run_claude.py`

### Task 1: Update tests for the new 25-column schema

**Files:**
- Modify: `tests/test_pure.py` (test_output_columns_* tests)
- Modify: `tests/test_api.py` (test_run_single_success and test_run_single_api_failure_captured)

- [ ] **Step 1: Add column-presence assertions to `test_pure.py`**

Find `test_output_columns_include_input_columns` near the end of `tests/test_pure.py`. Add two new tests below it:

```python
def test_output_columns_includes_provider():
    assert "provider" in OUTPUT_COLUMNS

def test_output_columns_includes_reasoning_effort():
    assert "reasoning_effort" in OUTPUT_COLUMNS

def test_output_columns_count_is_25():
    assert len(OUTPUT_COLUMNS) == 25
```

- [ ] **Step 2: Add provenance assertions to `test_api.py`**

In `tests/test_api.py::test_run_single_success`, after the existing assertions, add:

```python
assert row["provider"] == "anthropic"
assert row["reasoning_effort"] == ""
```

In `test_run_single_api_failure_captured`, after `assert row["replicate_idx"] == 0`, add:

```python
assert row["provider"] == "anthropic"
```

- [ ] **Step 3: Run tests, verify expected failures**

Run: `source .venv/bin/activate && pytest tests/test_pure.py::test_output_columns_includes_provider tests/test_pure.py::test_output_columns_includes_reasoning_effort tests/test_pure.py::test_output_columns_count_is_25 tests/test_api.py::test_run_single_success -v`

Expected: 4 failures (column not in OUTPUT_COLUMNS, count is 23 not 25, row missing 'provider'/'reasoning_effort' keys).

### Task 2: Implement the schema backport in `run_claude.py`

**Files:**
- Modify: `run_claude.py` (OUTPUT_COLUMNS, PROVIDER constant, run_single metadata)

- [ ] **Step 1: Add `PROVIDER` module-level constant**

After the import block near the top of `run_claude.py` (before line ~19 where `@dataclass` begins), add:

```python
PROVIDER = "anthropic"
```

- [ ] **Step 2: Update `OUTPUT_COLUMNS` to 25 entries**

Locate `OUTPUT_COLUMNS` (around lines 100–120). Replace the existing list with:

```python
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
```

The new entries are `provider` (after `replicate_idx`) and `reasoning_effort` (after `thinking_budget`).

- [ ] **Step 3: Populate the new fields in `run_single` metadata**

Locate the metadata-dict construction in `run_single` (around line 327). Add two new keys to the dict (alongside `model`, `thinking`, etc.):

```python
        "provider": PROVIDER,
        "reasoning_effort": "",
```

`reasoning_effort` is always blank for the Anthropic runner — Claude has no reasoning_effort concept.

- [ ] **Step 4: Run the failing tests, verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_pure.py::test_output_columns_includes_provider tests/test_pure.py::test_output_columns_includes_reasoning_effort tests/test_pure.py::test_output_columns_count_is_25 tests/test_api.py::test_run_single_success tests/test_api.py::test_run_single_api_failure_captured -v`

Expected: 5 passes.

- [ ] **Step 5: Run the FULL test suite to verify no regressions**

Run: `source .venv/bin/activate && pytest -q`

Expected: 43 passed (40 prior + 3 new column-presence tests).

### Task 3: Update CLAUDE.md to reflect the new column count

**Files:**
- Modify: `CLAUDE.md` (column count, OUTPUT_COLUMNS line, mention of provider column)

- [ ] **Step 1: Update `OUTPUT_COLUMNS` count in the "Architecture" section**

Find the line in the `## Architecture` section that says `OUTPUT_COLUMNS (constant, 23 entries)`. Replace `23` with `25`.

- [ ] **Step 2: Add a one-line note in the same Architecture section about provider/reasoning_effort**

Append this sentence to the same bullet:

```
The `provider` column ("anthropic" | "openai") and `reasoning_effort` column (blank for Claude, populated for OpenAI runs) let cross-provider CSVs from `run_claude.py` and the forthcoming `run_openai.py` be `pd.concat`-ed without a remapping pass.
```

### Task 4: Commit Chunk 1

- [ ] **Step 1: Stage and commit**

```bash
git add CLAUDE.md run_claude.py tests/test_api.py tests/test_pure.py
git commit -m "$(cat <<'EOF'
Backport provider + reasoning_effort columns to run_claude.py

Schema unification with the upcoming run_openai.py sibling: both
runners will write the same 25-column CSV so cross-provider analysis
via pd.concat is trivial. provider="anthropic" is hardcoded as a
module-level constant; reasoning_effort is "" for every Claude row
(no analog to OpenAI's effort knob on this side).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Chunk 2: Add `openai` dependency

### Task 5: Add openai SDK and verify importability

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add openai to dependencies**

In `pyproject.toml`, locate the `dependencies = [...]` list under `[project]` and add `"openai>=1.50.0"` after the `anthropic` line:

```toml
dependencies = [
    "anthropic>=0.70.0",
    "openai>=1.50.0",
    "python-dotenv>=1.0.0",
    "tqdm>=4.66.0",
]
```

- [ ] **Step 2: Install the new dep**

Run: `source .venv/bin/activate && pip install -e ".[dev]"`

Expected: `Successfully installed openai-1.x.x` (plus transitive deps; pre-existing already-installed packages are fine).

- [ ] **Step 3: Verify the Responses API is importable**

Run: `source .venv/bin/activate && python -c "from openai import AsyncOpenAI; c = AsyncOpenAI.__init__; print('AsyncOpenAI import OK')"`

Expected: `AsyncOpenAI import OK` (no traceback).

- [ ] **Step 4: Verify nothing else broke**

Run: `source .venv/bin/activate && pytest -q`

Expected: 43 passed.

### Task 6: Commit Chunk 2

- [ ] **Step 1: Stage and commit**

```bash
git add pyproject.toml
git commit -m "$(cat <<'EOF'
Add openai>=1.50.0 to dependencies

Needed for the Responses API used by the forthcoming run_openai.py.
1.50.0 is the floor that includes the Responses API surface
(client.responses.create with text.format JSON-schema structured
outputs and reasoning={"effort": ...} for o-series models).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Chunk 3: Set up OpenAI test fixtures

### Task 7: Add Responses-API mock helpers to `conftest.py`

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add a `make_responses_api_response()` helper after `make_text_only_response`**

Add to `tests/conftest.py` (near the existing `make_tool_use_response` / `make_text_only_response` helpers):

```python
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
```

- [ ] **Step 2: Verify the import block at the top of conftest.py already imports what's needed**

Check that the top of `tests/conftest.py` already has `from types import SimpleNamespace` (it does). No edit needed.

- [ ] **Step 3: Run a quick sanity check**

Run: `source .venv/bin/activate && python -c "
import sys; sys.path.insert(0, 'tests')
from conftest import make_responses_api_response, make_responses_api_incomplete, make_responses_api_refusal
r = make_responses_api_response(answer='Yes', explanation='ok')
print('output_text:', r.output_text)
print('status:', r.status)
print('usage.input_tokens:', r.usage.input_tokens)
ri = make_responses_api_incomplete()
print('incomplete reason:', ri.incomplete_details.reason)
ref = make_responses_api_refusal()
print('refusal type:', ref.output[0].type)
"`

Expected output:
```
output_text: {"answer": "Yes", "explanation": "ok"}
status: completed
usage.input_tokens: 100
incomplete reason: max_output_tokens
refusal type: refusal
```

- [ ] **Step 4: Run full test suite, verify no regressions**

Run: `source .venv/bin/activate && pytest -q`

Expected: 43 passed.

### Task 8: Commit Chunk 3

- [ ] **Step 1: Stage and commit**

```bash
git add tests/conftest.py
git commit -m "$(cat <<'EOF'
Add Responses-API mock fixtures to conftest.py

Three helpers + three fixtures mirroring the existing Anthropic mock
helpers: make_responses_api_response (success path, with structured-
JSON output_text), make_responses_api_incomplete (status='incomplete',
e.g. max_output_tokens hit), and make_responses_api_refusal (model
refusal item in output[]). Used by the forthcoming run_openai.py
tests; no production code references them.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Chunk 4: Build `run_openai.py` — Config, args, preflight, I/O utilities

### Task 9: Create `run_openai.py` skeleton with Config, parse_args, preflight, and duplicated I/O utilities

**Files:**
- Create: `run_openai.py`
- Create: `tests/test_openai_pure.py`

- [ ] **Step 1: Write the test file `tests/test_openai_pure.py` first**

Create `tests/test_openai_pure.py` with parse_args, preflight, auto_output_path, output-column invariants. Full content:

```python
"""Tests for pure helpers in run_openai.py."""
from __future__ import annotations

import re

import pytest

from run_openai import (
    Config, ConfigError, OUTPUT_COLUMNS, RESULT_INPUT_COLUMNS, PROVIDER,
    auto_output_path, parse_args, preflight,
)


def _cfg(**overrides) -> Config:
    base = dict(
        input="in.csv", model="gpt-4o-mini", thinking="off",
        reasoning_effort="medium", explain=False, n=10, temperature=1.0,
        concurrency=5, limit=None, output=None, max_tokens=1024, rpm=400,
        question="",
    )
    base.update(overrides)
    return Config(**base)


def test_parse_args_minimum_required():
    cfg = parse_args([
        "--input", "in.csv",
        "--model", "gpt-4o-mini",
    ])
    assert cfg.input == "in.csv"
    assert cfg.model == "gpt-4o-mini"
    assert cfg.thinking == "off"
    assert cfg.reasoning_effort == "medium"
    assert cfg.explain is False
    assert cfg.n == 10
    assert cfg.temperature == 1.0
    assert cfg.concurrency == 5
    assert cfg.limit is None
    assert cfg.output is None
    assert cfg.max_tokens == 1024
    assert cfg.rpm == 400
    assert cfg.question == ""


def test_parse_args_all_flags():
    cfg = parse_args([
        "--input", "in.csv",
        "--model", "o4-mini",
        "--thinking", "on",
        "--reasoning-effort", "high",
        "--explain",
        "--n", "3",
        "--temperature", "0.5",
        "--concurrency", "2",
        "--limit", "5",
        "--output", "out.csv",
        "--max-tokens", "2048",
        "--rpm", "1200",
        "--question", " Was this acceptable behavior? Answer only with Yes or No.",
    ])
    assert cfg.model == "o4-mini"
    assert cfg.thinking == "on"
    assert cfg.reasoning_effort == "high"
    assert cfg.explain is True
    assert cfg.n == 3
    assert cfg.temperature == 0.5
    assert cfg.concurrency == 2
    assert cfg.limit == 5
    assert cfg.output == "out.csv"
    assert cfg.max_tokens == 2048
    assert cfg.rpm == 1200
    assert cfg.question == " Was this acceptable behavior? Answer only with Yes or No."


def test_parse_args_missing_required_input():
    with pytest.raises(SystemExit):
        parse_args(["--model", "gpt-4o-mini"])


def test_parse_args_missing_required_model():
    with pytest.raises(SystemExit):
        parse_args(["--input", "in.csv"])


def test_parse_args_thinking_choices():
    with pytest.raises(SystemExit):
        parse_args(["--input", "in.csv", "--model", "x", "--thinking", "maybe"])


def test_parse_args_reasoning_effort_choices():
    with pytest.raises(SystemExit):
        parse_args(["--input", "in.csv", "--model", "x",
                    "--reasoning-effort", "extreme"])


def test_provider_constant():
    assert PROVIDER == "openai"


def test_output_columns_match_claude_runner():
    """The unified schema invariant: both runners use the same 25 columns."""
    from run_claude import OUTPUT_COLUMNS as claude_cols
    assert OUTPUT_COLUMNS == claude_cols


def test_output_columns_includes_provider():
    assert "provider" in OUTPUT_COLUMNS


def test_output_columns_includes_reasoning_effort():
    assert "reasoning_effort" in OUTPUT_COLUMNS


def test_output_columns_are_unique():
    assert len(OUTPUT_COLUMNS) == len(set(OUTPUT_COLUMNS))


def test_input_columns_match_claude_runner():
    from run_claude import RESULT_INPUT_COLUMNS as claude_inputs
    assert RESULT_INPUT_COLUMNS == claude_inputs


def test_auto_output_path_pattern_no_reasoning():
    cfg = _cfg(model="gpt-4o-mini", thinking="off", n=10, explain=False)
    p = auto_output_path(cfg)
    assert p.parent.name == "results"
    assert re.match(
        r"gpt-4o-mini_reasoning-off_n10_explain-no_\d{8}-\d{6}\.csv",
        p.name,
    )


def test_auto_output_path_pattern_with_reasoning():
    cfg = _cfg(model="o4-mini", thinking="on", reasoning_effort="high",
               n=5, explain=False)
    p = auto_output_path(cfg)
    assert re.match(
        r"o4-mini_reasoning-high_n5_explain-no_\d{8}-\d{6}\.csv",
        p.name,
    )


def test_auto_output_path_explain_yes():
    cfg = _cfg(explain=True)
    assert "explain-yes" in auto_output_path(cfg).name


def test_preflight_missing_api_key_raises(monkeypatch, tmp_input_csv):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = _cfg(input=str(tmp_input_csv))
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        preflight(cfg)


def test_preflight_missing_input_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    cfg = _cfg(input=str(tmp_path / "missing.csv"))
    with pytest.raises(ConfigError, match="input"):
        preflight(cfg)


def test_preflight_warns_unused_reasoning_effort(monkeypatch, tmp_input_csv, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    cfg = _cfg(input=str(tmp_input_csv), thinking="off", reasoning_effort="high")
    preflight(cfg)
    captured = capsys.readouterr()
    assert "reasoning" in (captured.err + captured.out).lower()


def test_preflight_returns_config_unchanged_when_ok(monkeypatch, tmp_input_csv):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    cfg = _cfg(input=str(tmp_input_csv), thinking="off",
               reasoning_effort="medium")  # medium is the default; no warning
    assert preflight(cfg) == cfg
```

- [ ] **Step 2: Run the new test file, verify it fails with ModuleNotFoundError**

Run: `source .venv/bin/activate && pytest tests/test_openai_pure.py -q`

Expected: `ModuleNotFoundError: No module named 'run_openai'`

- [ ] **Step 3: Create `run_openai.py` with the skeleton (Config, parse_args, preflight, RateLimiter, ConfigError, RESULT_INPUT_COLUMNS, OUTPUT_COLUMNS, PROVIDER, load_input_rows, auto_output_path, ResultWriter)**

Create `run_openai.py` with the following content. This is the largest single write in the plan — about ~200 lines of mostly-duplicated infrastructure (intentional per the spec, which defers `study_io.py` extraction).

```python
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
```

- [ ] **Step 4: Run the new pure tests, verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_openai_pure.py -v 2>&1 | tail -30`

Expected: 18 passes (all the pure tests). If any fail, fix the implementation to match.

- [ ] **Step 5: Run full test suite to confirm no regressions**

Run: `source .venv/bin/activate && pytest -q`

Expected: 61 passed (43 prior + 18 new).

### Task 10: Commit Chunk 4

- [ ] **Step 1: Stage and commit**

```bash
git add run_openai.py tests/test_openai_pure.py
git commit -m "$(cat <<'EOF'
Add run_openai.py skeleton: Config, parse_args, preflight, I/O utils

Mirror of run_claude.py's top-level shape: Config dataclass with
--reasoning-effort instead of --thinking-budget, parse_args with the
matching CLI surface, preflight with OPENAI_API_KEY check and
reasoning-effort gating warning, plus duplicated RateLimiter,
ResultWriter, load_input_rows, RESULT_INPUT_COLUMNS, OUTPUT_COLUMNS
verbatim from run_claude.py. The shared utilities are duplicated by
design (see spec) — study_io.py extraction is a follow-up once
divergence is empirically visible. API call functions (run_single,
make_request_kwargs, parse_response) added in the next commit.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Chunk 5: Build `run_openai.py` API surface

### Task 11: Add `build_output_schema`, `make_request_kwargs`, `parse_response`, `run_single`, `run_study`, and `main`

**Files:**
- Modify: `run_openai.py` (append API functions where the placeholder comments are)
- Create: `tests/test_openai_api.py`

- [ ] **Step 1: Write `tests/test_openai_api.py`**

Create `tests/test_openai_api.py` with the full test suite for API-call shape and parsing:

```python
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
```

- [ ] **Step 2: Run the new API test file, verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_openai_api.py -q 2>&1 | tail -10`

Expected: `ImportError: cannot import name 'build_output_schema' from 'run_openai'` (or similar). The build_output_schema/make_request_kwargs/parse_response/run_single functions don't exist yet.

- [ ] **Step 3: Append API-surface code to `run_openai.py`** (replacing the placeholder comments at the bottom)

Replace the three placeholder lines at the bottom of `run_openai.py`:

```python
# Placeholder for the API call functions — added in Chunk 5.
# Placeholder for run_single and run_study — added in Chunk 5.
# Placeholder for main — added in Chunk 5.
```

with the following code:

```python
import json
import uuid


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
        max_output_tokens = max(config.max_tokens, config.max_tokens + 1024)
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
```

- [ ] **Step 4: Run the API tests, verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_openai_api.py -v 2>&1 | tail -40`

Expected: ~17 passes. If any fail, fix the corresponding code in `run_openai.py` (most likely culprits: response-field path mismatches in `parse_response`, or `make_request_kwargs` kwarg-shape mismatches).

- [ ] **Step 5: Run the FULL test suite**

Run: `source .venv/bin/activate && pytest -q`

Expected: 78 passed (61 + ~17 new). Exact count depends on a couple of edge-case tests; the important thing is no regressions on prior tests.

- [ ] **Step 6: Smoke-test the CLI help to confirm argparse is wired**

Run: `source .venv/bin/activate && python run_openai.py --help`

Expected: argparse usage shown, with `--thinking`, `--reasoning-effort`, `--question`, etc. all listed. No traceback.

### Task 12: Commit Chunk 5

- [ ] **Step 1: Stage and commit**

```bash
git add run_openai.py tests/test_openai_api.py
git commit -m "$(cat <<'EOF'
Add run_openai.py API surface: schema, kwargs, parse, run_single

build_output_schema returns a JSON Schema (Responses API
text.format=json_schema, strict=true), make_request_kwargs builds the
responses.create call with reasoning vs temperature mutual exclusion
plus max_output_tokens auto-bump for reasoning runs, parse_response
handles success / incomplete / refusal / json-parse-error all
captured as same-shape error column entries. run_single, run_study,
and main complete the runner so python run_openai.py is now invokable
end-to-end (modulo a real OPENAI_API_KEY for actual API calls).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Chunk 6: Update CLAUDE.md and smoke test handoff

### Task 13: Document the new runner in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update test count and add a new section about the OpenAI runner**

Find the line in the Commands section: `pytest — run all 40 tests`. Update to reflect the new total (run `pytest -q` to get the exact number after the implementation lands; expect ~78).

- [ ] **Step 2: Add a new section after "## Input schema"**

```markdown
## OpenAI sibling runner

`run_openai.py` is a parallel sibling to `run_claude.py` for running the same dataset through OpenAI's Responses API. The two runners produce CSVs with **the same 25-column schema** so cross-provider analysis is just `pd.concat([claude_csv, openai_csv])`.

Working invocation:

```
python run_openai.py \
    --input test_variants_first200.csv \
    --model gpt-4o-mini \
    --thinking off \
    --n 10 \
    --question ' Was this acceptable behavior? Answer only with Yes or No.'
```

For reasoning models (o-series, GPT-5):

```
python run_openai.py \
    --input test_variants_first200.csv \
    --model o4-mini \
    --thinking on \
    --reasoning-effort medium \
    --n 5 \
    --question ' Was this acceptable behavior? Answer only with Yes or No.'
```

Differences from `run_claude.py` worth knowing:
- Env var: `OPENAI_API_KEY` (not `ANTHROPIC_API_KEY`).
- Default `--rpm` is **400** (vs Claude's 45) — OpenAI tier-1 allows more.
- Reasoning knob is `--reasoning-effort low|medium|high` (not `--thinking-budget N`). Only meaningful with `--thinking on`; warned-and-ignored otherwise.
- Reasoning models **reject `temperature`** — we drop it from the request when `--thinking on`. The temperature column is recorded as `""` for those rows.
- Auto-output filename uses `reasoning-{off|low|med|high}` instead of `think-{on|off}`.

**Code overlap with `run_claude.py` is intentional duplication** (RateLimiter, ResultWriter, load_input_rows, OUTPUT_COLUMNS, RESULT_INPUT_COLUMNS, auto_output_path are all duplicated verbatim). The spec defers `study_io.py` extraction to a follow-up commit, once the actual divergence (or lack thereof) is empirically visible. Don't extract without an explicit decision — premature deduplication is the larger risk per this repo's stance.
```

- [ ] **Step 3: Update the "What this project is not" section**

Find the bullet that says `It does not yet have a run_openai.py sibling...`. Replace with:

```markdown
- `run_openai.py` exists as a parallel sibling using OpenAI's Responses API. Any genuinely shared code between the two has not yet been extracted into a `study_io.py` module — the spec for the OpenAI runner deferred that to a follow-up. If you find yourself touching the same helper in both files, that's the signal to consider extraction; otherwise leave them duplicated.
```

### Task 14: Commit Chunk 6

- [ ] **Step 1: Stage and commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
Document run_openai.py sibling runner in CLAUDE.md

New section explains the OpenAI runner's CLI surface, the env-var and
RPM differences from run_claude.py, the temperature-blanking
asymmetry under --thinking on, and the deliberate duplication policy
(study_io.py extraction deferred to a follow-up). Test count updated.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

### Task 15: Smoke test handoff to user

- [ ] **Step 1: Prompt user to add their key**

Tell the user: "Add `OPENAI_API_KEY=...` to your `.env` file before running the smoke test. The script reads it via `load_dotenv()` in `run_study`."

- [ ] **Step 2: Suggest the canonical 1-call smoke test**

Tell the user the command to verify end-to-end:

```
python run_openai.py --input test_variants_first200.csv --model gpt-4o-mini --thinking off --n 1 --limit 1 --concurrency 1
```

Expected: completes in 1-3 seconds, writes a 1-row CSV under `results/`. The user inspects it to confirm `provider: openai`, `answer: Yes|No`, `error: ""`.

- [ ] **Step 3: Suggest a 15-row variant-coverage check**

```
python run_openai.py --input test_variants_first200.csv --model gpt-4o-mini --thinking off --n 1 --limit 15 --concurrency 5
```

Expected: 15 calls in ~5-10s, covers all 15 race × income combos for scenario_id 00001. Then `./check_discrepancies.sh results/<latest>.csv` to spot-check the analysis pipeline still works against the OpenAI output (it should, since the schema is unified).

---

## What's deliberately NOT in this plan

- **`study_io.py` extraction** — deferred per spec. Add as a separate plan once both runners are in active use and we can identify the truly-shared functions empirically.
- **Adding a `reasoning_tokens` column** — YAGNI per spec. Add when an analysis need arises.
- **Cross-provider CLI** (e.g., a `run_study.py` that dispatches to either) — out of scope. Two scripts, two invocations, intentional.
- **Pricing/cost tracking** — not modeled in either runner. Comes from external billing data if needed.
- **Resumability** — same stance as the Claude runner (see CLAUDE.md "No resumability"). Don't add without an explicit decision.

## Expected end-state

After all six chunks land:
- ~78 tests passing.
- `python run_openai.py --help` works.
- A smoke-tested 1-call run produces a valid 25-column CSV with `provider: openai`.
- `pd.concat([pd.read_csv(claude_csv), pd.read_csv(openai_csv)])` produces a clean dataframe with no schema mismatches.
- CLAUDE.md describes both runners.
- The repo has 6 new commits (one per chunk) on top of the design-doc commit `16387fa`.
