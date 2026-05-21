# Claude Moral-Variant Runner — Design

**Date:** 2026-05-21
**Status:** Approved (brainstorming phase complete)
**Author:** drafted with Claude Code during /init session

## Purpose

A Python script that processes the moral-scenario CSV (`moral_variant_binary_data.csv`) by sending each prompt to the Anthropic Claude API and recording structured Yes/No answers (optionally with explanations) into an output CSV. The dataset is designed to detect demographic bias in moral judgments: each base scenario appears in five demographic variants, so the analysis goal is to compare answer distributions across variants.

The script will be paired later with a parallel `run_openai.py`. Naming and shape are chosen to make that copy-and-adapt step straightforward, but no shared abstraction is built in v1.

## Inputs

A CSV with columns: `base_id`, `original_scenario_a`, `variant_description`, `full_binary_prompt`, `virtue`. Each row's `full_binary_prompt` is the message text sent to the model. The prompts already end with "Answer only with Yes or No." — left in place to preserve the authored prompt; the structured-output mechanism (not the natural-language instruction) is what enforces the answer format.

## CLI

Single configuration per invocation. Run the script multiple times with different flags to compare configurations.

```
python run_claude.py \
    --input moral_variant_binary_data.csv \
    --model claude-opus-4-7 \
    --thinking on|off \
    --thinking-budget 4096 \
    --explain \
    --n 10 \
    --temperature 1.0 \
    --concurrency 5 \
    --limit N \
    --output results/run_<auto>.csv
```

Flag semantics:

- `--input` (required): path to the input CSV.
- `--model` (required): Anthropic model ID. No default — the user must pick.
- `--thinking on|off` (default `off`): toggles extended thinking. When `on`, requires `temperature=1.0`.
- `--thinking-budget` (default `4096`): tokens for extended thinking. Ignored with warning if `--thinking off`.
- `--explain` (flag, default off): when set, the model must produce both an answer and an explanation. When unset, explanation is optional in the schema and stored as empty string.
- `--n` (default `10`): replicates per prompt.
- `--temperature` (default `1.0`): sampling temperature. Forced to `1.0` when `--thinking on` (warning emitted if user sets otherwise).
- `--concurrency` (default `5`): max in-flight API requests.
- `--limit` (default `None`): if set, processes only the first N rows of the input CSV. For smoke-testing config without burning calls.
- `--output` (default auto-generated): path to the output CSV. Auto pattern: `results/{model}_think-{on|off}_n{N}_explain-{yes|no}_{YYYYMMDD-HHMMSS}.csv`.

`ANTHROPIC_API_KEY` is loaded from `.env` via `python-dotenv`. The script fails loudly at startup if it is missing.

## Project layout

```
510_final_project/
├── moral_variant_binary_data.csv     # input (existing)
├── run_claude.py                      # this script
├── run_openai.py                      # future sibling, not in scope
├── pyproject.toml                     # dependencies
├── .env                               # ANTHROPIC_API_KEY, gitignored
├── .gitignore
├── results/                           # output CSVs land here
├── docs/superpowers/specs/            # design docs
└── README.md
```

## API call shape

A single tool is defined, and the model is forced to call it on every request:

```python
tools = [{
    "name": "submit_answer",
    "description": "Submit your answer to the moral scenario.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "enum": ["Yes", "No"],
                "description": "Your answer to the question."
            },
            "explanation": {
                "type": "string",
                "description": "Brief explanation of your reasoning."
            }
        },
        # "explanation" added to required list when --explain is set
        "required": ["answer"]
    }
}]
```

- Without `--thinking`: `tool_choice={"type": "tool", "name": "submit_answer", "disable_parallel_tool_use": True}` — strongest possible forcing, single `tool_use` block guaranteed.
- With `--thinking`: extended thinking does not allow `tool_choice: tool` (verify against current Anthropic docs at implementation time, as the constraint surface has shifted across SDK versions). Use `tool_choice={"type": "any", "disable_parallel_tool_use": True}` instead. Since only one tool is defined, the effect is equivalent.
- `messages=[{"role": "user", "content": full_binary_prompt}]`. No system prompt in v1.
- `max_tokens`: required by the Messages API. Default `1024`. When `--thinking on`, `max_tokens` must exceed `--thinking-budget` — script auto-bumps it to `thinking_budget + 1024` if the user set a value too small (with a warning).
- The expected response contains a `tool_use` content block with `name="submit_answer"` and `input={"answer": "Yes"|"No", "explanation": "..."}`. The script extracts this block; if `stop_reason != "tool_use"` it records the failure mode in the `stop_reason` and `error` columns rather than crashing.
- Tool schema mutation for `--explain`: at startup, if `--explain` is set, append `"explanation"` to the schema's `required` list before passing to the SDK. The same `tools` array is reused across all requests in the run.

**Methodology note:** with `--explain`, the answer and the explanation are produced in the same forward pass. The explanation is therefore best read as a post-hoc justification, not as the reasoning that led to the answer. When `--thinking on`, the model reasons inside the thinking block before emitting the tool call; the explanation field is still post-hoc summary, not the thinking itself.

## Output CSV schema

One row per replicate (long format). Header written immediately; rows stream-appended as calls complete (not necessarily in input order). `csv.DictWriter` with `quoting=csv.QUOTE_ALL` so explanations containing commas, quotes, and newlines are handled cleanly.

| Column | Source | Notes |
|---|---|---|
| `base_id` | input | preserved |
| `original_scenario_a` | input | preserved |
| `variant_description` | input | preserved |
| `full_binary_prompt` | input | preserved |
| `virtue` | input | preserved |
| `replicate_idx` | added | `0`..`n-1` |
| `model` | added | the `--model` value |
| `thinking` | added | `on` / `off` |
| `thinking_budget` | added | integer or empty when `thinking=off` |
| `temperature` | added | float |
| `explain_requested` | added | `True` / `False` |
| `answer` | response | `Yes` / `No`, or empty on error |
| `explanation` | response | text or empty |
| `stop_reason` | response | `tool_use` / `end_turn` / `max_tokens` / etc. |
| `input_tokens` | response | from `usage.input_tokens` |
| `output_tokens` | response | from `usage.output_tokens` |
| `cache_read_tokens` | response | from `usage.cache_read_input_tokens` (0 in v1) |
| `cache_creation_tokens` | response | from `usage.cache_creation_input_tokens` (0 in v1) |
| `latency_ms` | added | wall-clock per request |
| `timestamp_utc` | added | ISO-8601 UTC |
| `error` | added | empty unless the call failed after retries |
| `run_id` | added | one UUID per invocation, repeated on every row |

Config metadata is duplicated on every row so each CSV is self-describing and `pd.concat`-friendly across runs.

## Concurrency, retries, errors

- **Concurrency:** `asyncio` with `anthropic.AsyncAnthropic`. An `asyncio.Semaphore(concurrency)` caps in-flight requests. Tasks are awaited via `asyncio.as_completed()` so rows are written to disk as they finish — a mid-run crash leaves prior results intact on disk.
- **Retries:** the SDK's built-in retry (`max_retries=3`) handles 429s, 5xxs, and connection errors with exponential backoff. After exhaustion, the exception message is recorded in the `error` column and the run continues.
- **No crashes per row:** every exception from a single request is captured into the `error` field. The run only fails fast on configuration errors (missing API key, invalid model name surfaced as 400, missing input file).
- **No resumability in v1.** If a run crashes hard, re-run from scratch. Total wall-clock for the default config (50 prompts × 10 replicates × ~1–2s @ concurrency 5) is roughly 1–2 minutes. A `--resume <partial.csv>` flag can be added later if needed.

## Pre-flight checks

Before any API call:

1. Verify `ANTHROPIC_API_KEY` is set; fail with a clear message if not.
2. Read and validate the input CSV: required columns present; row count > 0.
3. Validate flag combinations:
   - `--thinking off` with `--thinking-budget` set → warn that budget is ignored.
   - `--thinking on` with `--temperature != 1.0` → warn and force 1.0.
4. Create `results/` if missing.
5. Print a one-line summary of the configuration and estimated call count (`min(rows, limit) × n` when `--limit` is set, otherwise `rows × n`).
6. Print the `run_id` to stdout so it can be recovered from terminal scrollback if the output filename is lost.
7. Initialize the output CSV with a header row.

A `tqdm` progress bar covers the request loop with live ETA.

## Dependencies

`pyproject.toml`:

- `anthropic >= 0.70.0`
- `python-dotenv`
- `tqdm`

Python 3.11+. `pandas` is not required by the script itself; downstream analysis can pull it in separately.

## Out of scope (v1)

- Prompt caching (token columns are captured anyway so we can see the hit rate when added later).
- Resumability of partial runs.
- System prompt customization.
- The OpenAI sibling script.
- A shared abstraction across providers. After both `run_claude.py` and `run_openai.py` exist, the genuinely shared parts (CSV I/O, CLI scaffolding) can be extracted into a small `study_io.py`.
- Analysis / plotting notebooks.
