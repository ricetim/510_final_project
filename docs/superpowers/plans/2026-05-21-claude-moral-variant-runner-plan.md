# Claude Moral-Variant Runner Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `run_claude.py` — a single-config-per-invocation Python script that runs the moral-variant CSV through the Anthropic Claude API with structured tool-use output and produces a long-format result CSV. CLI toggles for model, extended thinking, explanations, replicate count, concurrency, and a `--limit` for smoke tests.

**Architecture:** One script with internal functions organized by responsibility (config parsing, tool schema, response parsing, output path, CSV I/O, API call, async orchestrator). Top-level functions are importable from `tests/`. No package layout — the script lives at the repo root and is invoked as `python run_claude.py ...`.

**Tech Stack:** Python 3.10+, `anthropic` SDK (async), `python-dotenv`, `tqdm`, stdlib `csv` + `asyncio` + `argparse`, `pytest` for tests.

**Spec:** `docs/superpowers/specs/2026-05-21-claude-moral-variant-runner-design.md`

**Skills to reference during implementation:**
- @superpowers:test-driven-development — every behavior change goes test-first
- @superpowers:verification-before-completion — run the actual command and confirm the output before claiming a task done; never declare success on the basis of "the code looks right"
- @superpowers:systematic-debugging — if a test fails unexpectedly, find root cause; do not paper over with `try/except` or skips

---

## File Map

```
510_final_project/
├── moral_variant_binary_data.csv           # input, exists
├── run_claude.py                            # NEW — the script
├── pyproject.toml                           # NEW — deps, project metadata
├── .env.example                             # NEW — template for users
├── .env                                     # NEW (gitignored) — ANTHROPIC_API_KEY
├── README.md                                # NEW — quickstart + how to invoke
├── results/.gitkeep                         # NEW — preserves the dir in git
├── tests/
│   ├── __init__.py                          # NEW (empty)
│   ├── conftest.py                          # NEW — shared fixtures
│   ├── test_pure.py                         # NEW — CLI, tool schema, parser, paths
│   ├── test_io.py                           # NEW — input loader, output writer
│   └── test_api.py                          # NEW — mocked API call paths
└── docs/superpowers/{specs,plans}/          # exists
```

**Responsibilities of `run_claude.py` (alphabetical by function/class for navigation):**

- `Config` (dataclass) — parsed CLI args.
- `auto_output_path(config) -> Path` — pure: builds `results/<filename>.csv`.
- `build_tool_schema(explain: bool) -> dict` — pure: returns the tool definition.
- `load_input_rows(path, limit) -> list[dict]` — reads + validates the input CSV.
- `make_request_kwargs(config, prompt, tools)` — pure: builds the `messages.create` kwargs.
- `parse_args(argv=None) -> Config` — `argparse` wrapper.
- `parse_response(response, latency_ms) -> dict` — pure: extracts answer/explanation/usage/stop_reason from an API response.
- `preflight(config)` — env var, input file, flag-combo validation. Raises `ConfigError` on failure.
- `run_single(client, semaphore, config, tools, row, replicate_idx) -> dict` — one API call, returns a result row.
- `run_study(config) -> int` — async orchestrator. Returns exit code.
- `OUTPUT_COLUMNS` (constant) — ordered list of result-CSV columns.
- `RESULT_INPUT_COLUMNS` (constant) — ordered list of input columns we preserve.
- `main()` — entrypoint, calls `asyncio.run(run_study(parse_args()))`.

---

## Chunk 1: Scaffolding & pure helpers

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `README.md`
- Create: `results/.gitkeep`
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`

- [ ] **Step 1.1: Create `pyproject.toml`**

```toml
[project]
name = "moral-variant-runner"
version = "0.1.0"
description = "Run moral-variant CSV prompts through the Anthropic Claude API."
requires-python = ">=3.10"
dependencies = [
    "anthropic>=0.70.0",
    "python-dotenv>=1.0.0",
    "tqdm>=4.66.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 1.2: Create `.env.example`**

```
# Copy to .env and fill in your real key.
ANTHROPIC_API_KEY=sk-ant-...
```

- [ ] **Step 1.3: Create `results/.gitkeep`** (empty file)

- [ ] **Step 1.4: Create `tests/__init__.py`** (empty file)

- [ ] **Step 1.5: Create `tests/conftest.py`** with shared fixtures we'll fill in as we go:

```python
"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

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
```

- [ ] **Step 1.6: Create `README.md`** with a quickstart:

````markdown
# 510 Final Project — Moral-Variant Runner

Runs the moral-variant prompt CSV through the Anthropic Claude API and
records structured Yes/No answers (optionally with explanations) into an
output CSV. Designed to detect demographic bias by comparing answer
distributions across the five demographic variants per base scenario.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# edit .env and paste your ANTHROPIC_API_KEY
```

## Run

```bash
python run_claude.py \
    --input moral_variant_binary_data.csv \
    --model claude-opus-4-7 \
    --thinking off \
    --n 10
```

Output lands in `results/<auto-named>.csv`. See
`python run_claude.py --help` for all flags.

## Tests

```bash
pytest
```

## Design

See `docs/superpowers/specs/2026-05-21-claude-moral-variant-runner-design.md`.
````

- [ ] **Step 1.7: Create venv and install dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Expected: a green install with `anthropic`, `python-dotenv`, `tqdm`, `pytest`, `pytest-asyncio` in `pip list`.

- [ ] **Step 1.8: Verify pytest discovers an empty test suite**

```bash
source .venv/bin/activate && pytest
```

Expected: `no tests ran` (or similar). If it errors, fix `pyproject.toml` before moving on.

- [ ] **Step 1.9: Commit**

```bash
git add pyproject.toml .env.example README.md results/.gitkeep tests/
git commit -m "Scaffold project: deps, README, tests skeleton"
```

---

### Task 2: `Config` dataclass and `parse_args`

**Files:**
- Create: `run_claude.py`
- Create: `tests/test_pure.py`

- [ ] **Step 2.1: Write failing tests in `tests/test_pure.py`**

```python
"""Tests for pure helpers in run_claude.py."""
from __future__ import annotations

import pytest

from run_claude import Config, parse_args


def test_parse_args_minimum_required():
    cfg = parse_args([
        "--input", "in.csv",
        "--model", "claude-opus-4-7",
    ])
    assert cfg.input == "in.csv"
    assert cfg.model == "claude-opus-4-7"
    assert cfg.thinking == "off"           # default
    assert cfg.thinking_budget == 4096      # default
    assert cfg.explain is False
    assert cfg.n == 10
    assert cfg.temperature == 1.0
    assert cfg.concurrency == 5
    assert cfg.limit is None
    assert cfg.output is None
    assert cfg.max_tokens == 1024


def test_parse_args_all_flags():
    cfg = parse_args([
        "--input", "in.csv",
        "--model", "claude-sonnet-4-6",
        "--thinking", "on",
        "--thinking-budget", "8000",
        "--explain",
        "--n", "3",
        "--temperature", "0.5",
        "--concurrency", "2",
        "--limit", "5",
        "--output", "out.csv",
        "--max-tokens", "2048",
    ])
    assert cfg.model == "claude-sonnet-4-6"
    assert cfg.thinking == "on"
    assert cfg.thinking_budget == 8000
    assert cfg.explain is True
    assert cfg.n == 3
    assert cfg.temperature == 0.5
    assert cfg.concurrency == 2
    assert cfg.limit == 5
    assert cfg.output == "out.csv"
    assert cfg.max_tokens == 2048


def test_parse_args_missing_required_input(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--model", "claude-opus-4-7"])


def test_parse_args_missing_required_model(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--input", "in.csv"])


def test_parse_args_thinking_choices():
    with pytest.raises(SystemExit):
        parse_args(["--input", "in.csv", "--model", "x", "--thinking", "maybe"])
```

- [ ] **Step 2.2: Run tests — verify they fail**

```bash
source .venv/bin/activate && pytest tests/test_pure.py -v
```

Expected: `ModuleNotFoundError: No module named 'run_claude'` (or similar).

- [ ] **Step 2.3: Create `run_claude.py` with `Config` and `parse_args`**

```python
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
```

- [ ] **Step 2.4: Run tests — verify pass**

```bash
source .venv/bin/activate && pytest tests/test_pure.py -v
```

Expected: 5 passed.

- [ ] **Step 2.5: Commit**

```bash
git add run_claude.py tests/test_pure.py
git commit -m "Add Config dataclass and CLI arg parser"
```

---

### Task 3: `build_tool_schema`

**Files:**
- Modify: `run_claude.py`
- Modify: `tests/test_pure.py`

- [ ] **Step 3.1: Append failing tests to `tests/test_pure.py`**

```python
from run_claude import build_tool_schema


def test_tool_schema_explain_off():
    schema = build_tool_schema(explain=False)
    assert schema["name"] == "submit_answer"
    props = schema["input_schema"]["properties"]
    assert props["answer"]["enum"] == ["Yes", "No"]
    assert "explanation" in props
    assert schema["input_schema"]["required"] == ["answer"]


def test_tool_schema_explain_on():
    schema = build_tool_schema(explain=True)
    assert set(schema["input_schema"]["required"]) == {"answer", "explanation"}


def test_tool_schema_is_independent_per_call():
    """Mutating one returned schema must not affect a later call."""
    a = build_tool_schema(explain=False)
    a["input_schema"]["required"].append("explanation")
    b = build_tool_schema(explain=False)
    assert b["input_schema"]["required"] == ["answer"]
```

- [ ] **Step 3.2: Run — verify failure**

```bash
source .venv/bin/activate && pytest tests/test_pure.py::test_tool_schema_explain_off -v
```

Expected: `ImportError` or `AttributeError`.

- [ ] **Step 3.3: Implement `build_tool_schema` in `run_claude.py`**

```python
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
```

- [ ] **Step 3.4: Run — verify pass**

```bash
source .venv/bin/activate && pytest tests/test_pure.py -v
```

Expected: 8 passed.

- [ ] **Step 3.5: Commit**

```bash
git add run_claude.py tests/test_pure.py
git commit -m "Add build_tool_schema with explain toggle"
```

---

### Task 4: `parse_response`

**Files:**
- Modify: `run_claude.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_pure.py`

The Anthropic SDK returns objects with attributes like `response.content` (list of content blocks), `response.stop_reason`, `response.usage`. We model this with simple namespaces in tests rather than importing SDK types — keeps tests fast and SDK-version-tolerant.

- [ ] **Step 4.1: Add fixture helpers to `tests/conftest.py`**

```python
from types import SimpleNamespace


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
```

- [ ] **Step 4.2: Add failing tests to `tests/test_pure.py`**

```python
from run_claude import parse_response


def test_parse_response_tool_use(tool_use_response):
    r = tool_use_response(answer="Yes", explanation="ok",
                           input_tokens=120, output_tokens=15)
    parsed = parse_response(r, latency_ms=234)
    assert parsed["answer"] == "Yes"
    assert parsed["explanation"] == "ok"
    assert parsed["stop_reason"] == "tool_use"
    assert parsed["input_tokens"] == 120
    assert parsed["output_tokens"] == 15
    assert parsed["cache_read_tokens"] == 0
    assert parsed["cache_creation_tokens"] == 0
    assert parsed["latency_ms"] == 234
    assert parsed["error"] == ""


def test_parse_response_text_only_no_tool_call(text_only_response):
    r = text_only_response(text="Hmm", stop_reason="end_turn")
    parsed = parse_response(r, latency_ms=50)
    assert parsed["answer"] == ""
    assert parsed["explanation"] == ""
    assert parsed["stop_reason"] == "end_turn"
    assert "no tool_use block" in parsed["error"].lower()


def test_parse_response_missing_explanation(tool_use_response):
    """If the model omits explanation, store empty string, not None."""
    r = tool_use_response(answer="No", explanation=None)
    # Simulate model omitting the key entirely
    r.content[0].input = {"answer": "No"}
    parsed = parse_response(r, latency_ms=10)
    assert parsed["answer"] == "No"
    assert parsed["explanation"] == ""
```

- [ ] **Step 4.3: Run — verify failure**

- [ ] **Step 4.4: Implement `parse_response` in `run_claude.py`**

```python
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
```

- [ ] **Step 4.5: Run — verify pass**

```bash
source .venv/bin/activate && pytest tests/test_pure.py -v
```

- [ ] **Step 4.6: Commit**

```bash
git add run_claude.py tests/test_pure.py tests/conftest.py
git commit -m "Add parse_response for tool-use and fallback paths"
```

---

### Task 5: `auto_output_path` and column constants

**Files:**
- Modify: `run_claude.py`
- Modify: `tests/test_pure.py`

- [ ] **Step 5.1: Add failing tests to `tests/test_pure.py`**

```python
import re

from run_claude import (
    Config, auto_output_path, OUTPUT_COLUMNS, RESULT_INPUT_COLUMNS,
)


def _cfg(**overrides) -> Config:
    base = dict(
        input="in.csv", model="claude-opus-4-7", thinking="off",
        thinking_budget=4096, explain=False, n=10, temperature=1.0,
        concurrency=5, limit=None, output=None, max_tokens=1024,
    )
    base.update(overrides)
    return Config(**base)


def test_auto_output_path_pattern():
    cfg = _cfg(model="claude-opus-4-7", thinking="off", n=10, explain=False)
    p = auto_output_path(cfg)
    assert p.parent.name == "results"
    assert re.match(
        r"claude-opus-4-7_think-off_n10_explain-no_\d{8}-\d{6}\.csv",
        p.name,
    )


def test_auto_output_path_explain_yes():
    cfg = _cfg(explain=True)
    assert "explain-yes" in auto_output_path(cfg).name


def test_output_columns_include_input_columns():
    for col in RESULT_INPUT_COLUMNS:
        assert col in OUTPUT_COLUMNS


def test_output_columns_are_unique():
    assert len(OUTPUT_COLUMNS) == len(set(OUTPUT_COLUMNS))
```

- [ ] **Step 5.2: Run — verify failure**

- [ ] **Step 5.3: Implement in `run_claude.py`**

```python
from datetime import datetime, timezone
from pathlib import Path


RESULT_INPUT_COLUMNS: list[str] = [
    "base_id",
    "original_scenario_a",
    "variant_description",
    "full_binary_prompt",
    "virtue",
]


OUTPUT_COLUMNS: list[str] = [
    *RESULT_INPUT_COLUMNS,
    "replicate_idx",
    "model",
    "thinking",
    "thinking_budget",
    "temperature",
    "explain_requested",
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


def auto_output_path(config: Config) -> Path:
    explain_label = "yes" if config.explain else "no"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = (
        f"{config.model}_think-{config.thinking}_n{config.n}"
        f"_explain-{explain_label}_{stamp}.csv"
    )
    return Path("results") / name
```

- [ ] **Step 5.4: Run — verify pass**

- [ ] **Step 5.5: Commit**

```bash
git add run_claude.py tests/test_pure.py
git commit -m "Add OUTPUT_COLUMNS constants and auto_output_path"
```

---

### Task 6 — Chunk 1 review

Dispatch the plan-document-reviewer for Chunk 1 (Tasks 1–5) once it lands in the repo. Continue to Chunk 2 only after approval (or after the reviewer's feedback is incorporated).

---

## Chunk 2: CSV I/O and API call layer

### Task 7: `load_input_rows`

**Files:**
- Modify: `run_claude.py`
- Create: `tests/test_io.py`

- [ ] **Step 7.1: Write failing tests in `tests/test_io.py`**

```python
"""Tests for CSV I/O helpers."""
from __future__ import annotations

from pathlib import Path

import pytest

from run_claude import load_input_rows


def test_load_input_rows_basic(tmp_input_csv: Path):
    rows = load_input_rows(str(tmp_input_csv), limit=None)
    assert len(rows) == 3
    assert rows[0]["base_id"] == "PRU-001"
    assert "Answer only with Yes or No." in rows[0]["full_binary_prompt"]
    assert rows[2]["virtue"] == "Justice"


def test_load_input_rows_respects_limit(tmp_input_csv: Path):
    rows = load_input_rows(str(tmp_input_csv), limit=2)
    assert len(rows) == 2
    assert rows[1]["base_id"] == "PRU-001"


def test_load_input_rows_missing_column_raises(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    bad.write_text("base_id,full_binary_prompt\nX,hello\n")
    with pytest.raises(ValueError, match="missing required column"):
        load_input_rows(str(bad), limit=None)


def test_load_input_rows_empty_raises(tmp_path: Path):
    bad = tmp_path / "empty.csv"
    bad.write_text("base_id,original_scenario_a,variant_description,full_binary_prompt,virtue\n")
    with pytest.raises(ValueError, match="no data rows"):
        load_input_rows(str(bad), limit=None)


def test_load_input_rows_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_input_rows(str(tmp_path / "does-not-exist.csv"), limit=None)
```

- [ ] **Step 7.2: Run — verify failure**

- [ ] **Step 7.3: Implement `load_input_rows` in `run_claude.py`**

```python
import csv


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
```

- [ ] **Step 7.4: Run — verify pass**

- [ ] **Step 7.5: Commit**

```bash
git add run_claude.py tests/test_io.py
git commit -m "Add load_input_rows with schema validation"
```

---

### Task 8: Streaming output writer

**Files:**
- Modify: `run_claude.py`
- Modify: `tests/test_io.py`

The writer is a small class so the open file handle and `csv.DictWriter` live together with a cleanup `close()`. Use `QUOTE_ALL` so explanations with commas/newlines round-trip cleanly.

- [ ] **Step 8.1: Add failing tests to `tests/test_io.py`**

```python
import csv as _csv

from run_claude import OUTPUT_COLUMNS, ResultWriter


def test_result_writer_writes_header(tmp_path):
    out = tmp_path / "out.csv"
    w = ResultWriter(out)
    w.close()
    with out.open() as f:
        header = next(_csv.reader(f))
    assert header == OUTPUT_COLUMNS


def test_result_writer_appends_rows(tmp_path):
    out = tmp_path / "out.csv"
    w = ResultWriter(out)
    row = {c: "" for c in OUTPUT_COLUMNS}
    row.update({"base_id": "X", "answer": "Yes", "explanation": "with, comma"})
    w.write(row)
    w.close()
    with out.open() as f:
        reader = _csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["base_id"] == "X"
    assert rows[0]["answer"] == "Yes"
    assert rows[0]["explanation"] == "with, comma"


def test_result_writer_ignores_extra_keys(tmp_path):
    out = tmp_path / "out.csv"
    w = ResultWriter(out)
    row = {c: "" for c in OUTPUT_COLUMNS}
    row["unknown_key"] = "should not crash"
    w.write(row)
    w.close()
```

- [ ] **Step 8.2: Run — verify failure**

- [ ] **Step 8.3: Implement `ResultWriter` in `run_claude.py`**

```python
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
```

- [ ] **Step 8.4: Run — verify pass**

- [ ] **Step 8.5: Commit**

```bash
git add run_claude.py tests/test_io.py
git commit -m "Add ResultWriter for streaming CSV output"
```

---

### Task 9: `make_request_kwargs`

**Files:**
- Modify: `run_claude.py`
- Create: `tests/test_api.py`

This builds the kwargs dict passed to `client.messages.create(...)`. Separating it from the actual call makes it trivially testable.

- [ ] **Step 9.1: Write failing tests in `tests/test_api.py`**

```python
"""Tests for API call construction and execution (mocked)."""
from __future__ import annotations

import pytest

from run_claude import Config, build_tool_schema, make_request_kwargs


def _cfg(**overrides) -> Config:
    base = dict(
        input="in.csv", model="claude-opus-4-7", thinking="off",
        thinking_budget=4096, explain=False, n=10, temperature=1.0,
        concurrency=5, limit=None, output=None, max_tokens=1024,
    )
    base.update(overrides)
    return Config(**base)


def test_make_request_kwargs_no_thinking():
    cfg = _cfg(thinking="off", model="claude-opus-4-7", temperature=0.7)
    tools = [build_tool_schema(explain=False)]
    kw = make_request_kwargs(cfg, prompt="Hello?", tools=tools)
    assert kw["model"] == "claude-opus-4-7"
    assert kw["temperature"] == 0.7
    assert kw["max_tokens"] == 1024
    assert kw["messages"] == [{"role": "user", "content": "Hello?"}]
    assert kw["tools"] == tools
    assert kw["tool_choice"] == {
        "type": "tool", "name": "submit_answer", "disable_parallel_tool_use": True,
    }
    assert "thinking" not in kw


def test_make_request_kwargs_with_thinking():
    cfg = _cfg(thinking="on", thinking_budget=4096, temperature=1.0, max_tokens=1024)
    tools = [build_tool_schema(explain=False)]
    kw = make_request_kwargs(cfg, prompt="Hello?", tools=tools)
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert kw["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}
    # max_tokens must exceed thinking_budget; auto-bump applies
    assert kw["max_tokens"] >= 4096 + 1024


def test_make_request_kwargs_auto_bumps_max_tokens():
    cfg = _cfg(thinking="on", thinking_budget=8000, max_tokens=1024)
    tools = [build_tool_schema(explain=False)]
    kw = make_request_kwargs(cfg, prompt="x", tools=tools)
    assert kw["max_tokens"] >= 8000 + 1024
```

- [ ] **Step 9.2: Run — verify failure**

- [ ] **Step 9.3: Implement `make_request_kwargs` in `run_claude.py`**

```python
def make_request_kwargs(config: Config, prompt: str, tools: list[dict]) -> dict:
    """Build kwargs for client.messages.create()."""
    thinking_on = config.thinking == "on"
    if thinking_on:
        max_tokens = max(config.max_tokens, config.thinking_budget + 1024)
        tool_choice = {"type": "any", "disable_parallel_tool_use": True}
    else:
        max_tokens = config.max_tokens
        tool_choice = {
            "type": "tool",
            "name": "submit_answer",
            "disable_parallel_tool_use": True,
        }
    kw: dict = {
        "model": config.model,
        "max_tokens": max_tokens,
        "temperature": config.temperature,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "tool_choice": tool_choice,
    }
    if thinking_on:
        kw["thinking"] = {
            "type": "enabled",
            "budget_tokens": config.thinking_budget,
        }
    return kw
```

- [ ] **Step 9.4: Run — verify pass**

- [ ] **Step 9.5: Commit**

```bash
git add run_claude.py tests/test_api.py
git commit -m "Add make_request_kwargs with thinking + tool_choice handling"
```

---

### Task 10: `run_single` with mocked client

**Files:**
- Modify: `run_claude.py`
- Modify: `tests/test_api.py`

`run_single` is the one place we actually `await` the SDK. Mocking it directly is straightforward — we inject a fake client whose `.messages.create()` is an `AsyncMock`.

- [ ] **Step 10.1: Add failing tests to `tests/test_api.py`**

```python
import asyncio
from unittest.mock import AsyncMock

from run_claude import build_tool_schema, run_single


@pytest.fixture
def input_row() -> dict:
    return {
        "base_id": "PRU-001",
        "original_scenario_a": "orig",
        "variant_description": "variant 1",
        "full_binary_prompt": "Would you do X? Answer only with Yes or No.",
        "virtue": "Prudence",
    }


def _fake_client(response):
    client = AsyncMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


async def test_run_single_success(input_row, tool_use_response):
    client = _fake_client(tool_use_response(answer="Yes", explanation="ok"))
    cfg = _cfg()
    sem = asyncio.Semaphore(1)
    tools = [build_tool_schema(explain=False)]
    row = await run_single(client, sem, cfg, tools, input_row, replicate_idx=3,
                            run_id="abc-123")
    assert row["answer"] == "Yes"
    assert row["explanation"] == "ok"
    assert row["base_id"] == "PRU-001"
    assert row["replicate_idx"] == 3
    assert row["model"] == cfg.model
    assert row["thinking"] == "off"
    assert row["thinking_budget"] == ""           # blanked when thinking=off
    assert row["temperature"] == cfg.temperature
    assert row["explain_requested"] is False
    assert row["run_id"] == "abc-123"
    assert row["error"] == ""
    assert row["timestamp_utc"]                    # non-empty


async def test_run_single_with_thinking_records_budget(input_row, tool_use_response):
    client = _fake_client(tool_use_response())
    cfg = _cfg(thinking="on", thinking_budget=2048)
    sem = asyncio.Semaphore(1)
    tools = [build_tool_schema(explain=False)]
    row = await run_single(client, sem, cfg, tools, input_row, 0, "rid")
    assert row["thinking"] == "on"
    assert row["thinking_budget"] == 2048


async def test_run_single_api_failure_captured(input_row):
    client = AsyncMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("boom"))
    cfg = _cfg()
    sem = asyncio.Semaphore(1)
    tools = [build_tool_schema(explain=False)]
    row = await run_single(client, sem, cfg, tools, input_row, 0, "rid")
    assert row["answer"] == ""
    assert "boom" in row["error"]
    assert row["stop_reason"] == ""
    # Row still has all metadata columns:
    assert row["base_id"] == "PRU-001"
    assert row["replicate_idx"] == 0


async def test_run_single_no_tool_use_block(input_row, text_only_response):
    client = _fake_client(text_only_response())
    cfg = _cfg()
    sem = asyncio.Semaphore(1)
    tools = [build_tool_schema(explain=False)]
    row = await run_single(client, sem, cfg, tools, input_row, 0, "rid")
    assert row["answer"] == ""
    assert "no tool_use block" in row["error"].lower()
    assert row["stop_reason"] == "end_turn"
```

- [ ] **Step 10.2: Run — verify failure**

- [ ] **Step 10.3: Implement `run_single` in `run_claude.py`**

```python
import asyncio
import time
import uuid


async def run_single(
    client,
    semaphore: asyncio.Semaphore,
    config: Config,
    tools: list[dict],
    row: dict,
    replicate_idx: int,
    run_id: str,
) -> dict:
    """Issue one API call for one (row, replicate). Always returns a result row."""
    kwargs = make_request_kwargs(config, row["full_binary_prompt"], tools)
    metadata = {
        **{c: row.get(c, "") for c in RESULT_INPUT_COLUMNS},
        "replicate_idx": replicate_idx,
        "model": config.model,
        "thinking": config.thinking,
        "thinking_budget": config.thinking_budget if config.thinking == "on" else "",
        "temperature": config.temperature,
        "explain_requested": config.explain,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_id,
    }
    async with semaphore:
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
```

- [ ] **Step 10.4: Run — verify pass**

```bash
source .venv/bin/activate && pytest tests/test_api.py -v
```

- [ ] **Step 10.5: Commit**

```bash
git add run_claude.py tests/test_api.py
git commit -m "Add run_single: one mocked-API call per (row, replicate)"
```

---

### Task 11 — Chunk 2 review

Dispatch the plan-document-reviewer for Chunk 2 (Tasks 7–10). Address feedback before moving to Chunk 3.

---

## Chunk 3: Pre-flight, orchestrator, and main

### Task 12: `preflight`

**Files:**
- Modify: `run_claude.py`
- Modify: `tests/test_pure.py`

- [ ] **Step 12.1: Add failing tests to `tests/test_pure.py`**

```python
from run_claude import ConfigError, preflight


def test_preflight_missing_api_key_raises(monkeypatch, tmp_input_csv):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = _cfg(input=str(tmp_input_csv))
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        preflight(cfg)


def test_preflight_missing_input_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = _cfg(input=str(tmp_path / "missing.csv"))
    with pytest.raises(ConfigError, match="input"):
        preflight(cfg)


def test_preflight_temperature_forced_when_thinking_on(monkeypatch, tmp_input_csv, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = _cfg(input=str(tmp_input_csv), thinking="on", temperature=0.5)
    out = preflight(cfg)
    assert out.temperature == 1.0
    captured = capsys.readouterr()
    assert "temperature" in captured.err.lower() or "temperature" in captured.out.lower()


def test_preflight_warns_unused_budget(monkeypatch, tmp_input_csv, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = _cfg(input=str(tmp_input_csv), thinking="off", thinking_budget=9000)
    preflight(cfg)
    captured = capsys.readouterr()
    assert "ignored" in (captured.err + captured.out).lower()


def test_preflight_returns_config_unchanged_when_ok(monkeypatch, tmp_input_csv):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = _cfg(input=str(tmp_input_csv), thinking="off")
    assert preflight(cfg) == cfg
```

- [ ] **Step 12.2: Run — verify failure**

- [ ] **Step 12.3: Implement `preflight` in `run_claude.py`**

```python
import os
import sys
from dataclasses import replace


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
```

Note the test asserts the default `thinking_budget == 4096` is *not* treated as "user-set", so the warning only fires when the user explicitly chose a non-default value. If you'd rather warn whenever `--thinking off` and the user didn't actively suppress the warning, you can refine later; the current test only triggers on non-default budget.

- [ ] **Step 12.4: Run — verify pass**

- [ ] **Step 12.5: Commit**

```bash
git add run_claude.py tests/test_pure.py
git commit -m "Add preflight: env, file, and flag-combo validation"
```

---

### Task 13: `run_study` orchestrator + `main`

**Files:**
- Modify: `run_claude.py`

This is the only piece without dedicated unit tests — the orchestration is mostly plumbing (`asyncio.gather`, `tqdm`, opening the writer). The end-to-end smoke test in Task 14 covers it.

- [ ] **Step 13.1: Implement `run_study` and `main` in `run_claude.py`**

```python
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio


async def run_study(config: Config) -> int:
    """Top-level orchestrator. Returns process exit code."""
    load_dotenv()  # populate ANTHROPIC_API_KEY from .env if present
    try:
        config = preflight(config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rows = load_input_rows(config.input, config.limit)
    tools = [build_tool_schema(explain=config.explain)]
    output_path = Path(config.output) if config.output else auto_output_path(config)
    run_id = str(uuid.uuid4())

    n_calls = len(rows) * config.n
    print(
        f"run_id={run_id}  output={output_path}  "
        f"calls={n_calls} ({len(rows)} rows x {config.n} replicates)",
        file=sys.stderr,
    )

    # Import the SDK lazily so unit tests don't require it to be importable.
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(max_retries=3)
    semaphore = asyncio.Semaphore(config.concurrency)

    tasks = [
        asyncio.create_task(
            run_single(client, semaphore, config, tools, row, replicate_idx, run_id)
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

- [ ] **Step 13.2: Verify `python run_claude.py --help` prints sensible help**

```bash
source .venv/bin/activate && python run_claude.py --help
```

Expected: argparse help text listing every flag.

- [ ] **Step 13.3: Run the full test suite — everything should still pass**

```bash
source .venv/bin/activate && pytest
```

Expected: all tests green.

- [ ] **Step 13.4: Commit**

```bash
git add run_claude.py
git commit -m "Add run_study orchestrator and main entrypoint"
```

---

### Task 14: End-to-end smoke test against the real API

This is a manual verification step. Use `--limit 1 --n 1` to spend ~one API call.

- [ ] **Step 14.1: Ensure `.env` contains a real `ANTHROPIC_API_KEY`**

`cat .env` (or check via `printenv`). Do NOT commit `.env`.

- [ ] **Step 14.2: Smoke test — thinking off, no explain**

```bash
source .venv/bin/activate && \
python run_claude.py \
    --input moral_variant_binary_data.csv \
    --model claude-haiku-4-5-20251001 \
    --thinking off \
    --n 1 \
    --limit 1 \
    --concurrency 1
```

Expected:
- Stdout/stderr shows `run_id=...`, `output=results/<auto>.csv`, `calls=1`
- A single CSV row appears in the output file with a non-empty `answer` field (Yes or No) and `error == ""`
- `stop_reason == "tool_use"`

(Use Haiku for the smoke test — cheapest model.)

- [ ] **Step 14.3: Smoke test — thinking on, explain on**

```bash
python run_claude.py \
    --input moral_variant_binary_data.csv \
    --model claude-sonnet-4-6 \
    --thinking on \
    --thinking-budget 2048 \
    --explain \
    --n 1 \
    --limit 1 \
    --concurrency 1
```

Expected: a single row with both `answer` and `explanation` populated, `thinking == "on"`, `thinking_budget == 2048`, and `output_tokens` noticeably higher than the previous run.

- [ ] **Step 14.4: If the thinking + tool_choice constraint has changed since this plan was written**

If the call errors with something like "tool_choice 'any' not allowed with thinking" or "tool_choice 'tool' required", consult the current Anthropic Messages API docs and adjust the `tool_choice` branch in `make_request_kwargs`. Add a regression test capturing whatever the correct combination is.

- [ ] **Step 14.5: Commit any fixes from the smoke test**

```bash
git add run_claude.py tests/
git commit -m "Smoke-test fixes from real-API verification"
```

---

### Task 15: Push to GitHub

- [ ] **Step 15.1: Push all commits**

```bash
git push
```

- [ ] **Step 15.2: Verify on GitHub**

`gh repo view --web` (or visit https://github.com/ricetim/510_final_project) and confirm the latest commit is visible.

---

### Task 16 — Chunk 3 review

Dispatch the plan-document-reviewer for Chunk 3 (Tasks 12–15). Address feedback if any.

---

## Out of scope (deferred, per spec)

- Prompt caching (token-count columns are captured so a hit rate is observable when added later)
- Resumability via `--resume <partial.csv>`
- `run_openai.py` sibling
- Extraction of shared `study_io.py` after both providers exist
- Analysis / plotting notebooks
