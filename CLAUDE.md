# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A single-script Python research tool that runs the prompts in `moral_variant_binary_data.csv` through the Anthropic Claude API and records structured Yes/No answers (optionally with explanations) into a long-format CSV. The dataset has 50 prompts spanning five demographic variants × ten base scenarios across four virtues; the analysis goal is to detect demographic bias in moral judgments by comparing answer distributions across variants.

The full design is in `docs/superpowers/specs/2026-05-21-claude-moral-variant-runner-design.md`. The implementation plan is in `docs/superpowers/plans/2026-05-21-claude-moral-variant-runner-plan.md`. Both are the source of truth — read them before making non-trivial changes.

## Commands

Always activate the venv first: `source .venv/bin/activate`.

- `pytest` — run all 36 tests (no real API calls; everything is mocked via `SimpleNamespace` fixtures in `tests/conftest.py`).
- `pytest tests/test_pure.py::test_name -v` — run a single test by name.
- `python run_claude.py --help` — list every CLI flag and default.
- A working invocation:
  ```
  python run_claude.py \
      --input moral_variant_binary_data.csv \
      --model claude-haiku-4-5-20251001 \
      --thinking off \
      --n 10
  ```
- Smoke-test a config without burning credits: add `--limit 1 --n 1 --concurrency 1`.

API key lives in `.env` (gitignored). Output CSVs land in `results/` and are gitignored — commit summary tables only.

## Architecture

`run_claude.py` is intentionally a single file with module-level functions so `tests/` can import each unit. There is no package, no `src/`, no abstractions across providers. A future `run_openai.py` will be a parallel sibling, not a subclass — the SDKs differ enough (tool schema shape, thinking semantics) that copy-and-adapt beats a forced abstraction.

Functions in `run_claude.py`, by responsibility:

- `parse_args` → `Config` (frozen dataclass) — argparse, no defaults that depend on the environment.
- `preflight(config)` → corrected `Config` — validates `ANTHROPIC_API_KEY`, input file existence, and flag combos. Raises `ConfigError` on hard failure; corrects/warns on soft issues.
- `load_input_rows(path, limit)` → `list[dict]` — reads + validates input CSV columns.
- `build_tool_schema(explain)` → tool definition for forced structured output. **When `explain` is False the `explanation` field is omitted from `properties` entirely** so the model cannot volunteer one and contaminate runs intended to capture answers only.
- `make_request_kwargs(config, prompt, tools)` → `messages.create` kwargs. Handles the thinking-vs-no-thinking branch (see Gotchas).
- `run_single(client, semaphore, config, tools, row, replicate_idx, run_id)` (async) → result row. The only function that awaits the SDK. Captures all exceptions into the `error` column so one bad call doesn't kill the run.
- `parse_response(response, latency_ms)` → extracts `answer` / `explanation` / token counts / `stop_reason`. Records `error: "no tool_use block in response"` if the model returned text instead of calling the tool.
- `ResultWriter(path)` — streaming `csv.DictWriter` with `QUOTE_ALL` and `flush()` after every row for crash safety.
- `run_study(config)` (async) — the orchestrator. Loads dotenv, runs preflight, lazy-imports `AsyncAnthropic`, creates tasks for `len(rows) × config.n`, bounds concurrency with `asyncio.Semaphore`, streams rows via `tqdm_asyncio.as_completed` so a mid-run crash leaves prior rows on disk.
- `OUTPUT_COLUMNS` (constant, 22 entries) — the result-CSV schema. Config metadata is duplicated on every row so each file is self-describing for `pd.concat` across runs.

## Gotchas worth remembering

- **Extended thinking + forced tool use is rejected by the API.** With `--thinking on`, `tool_choice` must be `{"type": "auto"}` (not `"tool"` or `"any"`), so the model *may* return plain text. We compensate by appending `THINKING_TOOL_INSTRUCTION` to the prompt. In practice the model always calls the tool, but `parse_response` records `error: "no tool_use block in response"` on the rare miss. After any thinking-on run, sanity-check `pd.read_csv(out).query('error != ""')` — if non-empty, you have a methodology issue to address before analysis.
- **Output path is anchored to the script's directory, not CWD.** `auto_output_path` uses `Path(__file__).resolve().parent / "results"` so invocations from `~/` still land in the project's `results/`.
- **One config per invocation, by design.** No sweeps; the auto-generated filename `{model}_think-{on|off}_n{N}_explain-{yes|no}_{YYYYMMDD-HHMMSS}.csv` is meant to make shell-loop sweeps identifiable. If you find yourself reaching for sweep support, push back — the spec deliberately deferred it.
- **No resumability.** A full run is ~500 calls / ~1–2 minutes at default concurrency; cheaper to re-run than to maintain a `--resume` path. Don't add one without a real need.
- **`pyproject.toml` is for `pip install -e ".[dev]"`, not Poetry.** System Python is 3.10; `requires-python = ">=3.10"`. Don't bump to 3.11 without a real reason — nothing in the script uses 3.11+ features.

## When changing the schema

If you touch `RESULT_INPUT_COLUMNS` or `OUTPUT_COLUMNS`, the analysis side (downstream notebooks the user hasn't written yet) will break silently. Update both constants together and run the full test suite — `test_output_columns_include_input_columns` and the `ResultWriter` tests will catch most mismatches, but a column rename will pass tests while breaking analysis. Flag the change explicitly.

## What this project is not

- It is not a generic LLM eval framework. Don't add abstractions for "different datasets" or "different providers" until a second one actually exists.
- It is not a production service. No retry queues, no observability stack, no auth layer. The SDK's built-in `max_retries=3` is the entire reliability story.
- It does not yet have a `run_openai.py` sibling — the user intends to add one, at which point any genuinely shared code can be extracted into a small `study_io.py`. Until then, premature deduplication is the larger risk.
