# OpenAI Moral-Variant Runner — Design

**Date:** 2026-05-25
**Status:** Approved (brainstorming phase complete)
**Author:** drafted with Claude Code during the OpenAI-runner brainstorm session

## Purpose

Add a parallel sibling to `run_claude.py` that processes the same moral-scenario CSV via the OpenAI API. The two runners write output CSVs with an identical column schema so cross-provider comparison via `pd.concat` is trivial.

This sibling was planned for in the original Claude runner spec (2026-05-21): "A future `run_openai.py` will be a parallel sibling, not a subclass — the SDKs differ enough that copy-and-adapt beats a forced abstraction." This design honors that — building `run_openai.py` as a copy-and-adapt port, deferring any `study_io.py` extraction to a follow-up commit once the genuine overlap is empirically visible.

## Decisions

| Question | Choice | Reasoning |
|---|---|---|
| Which model classes? | **Chat + reasoning models (both)** | User wants the freedom to use gpt-4o-mini for cheap runs and o-series / GPT-5 for thinking-parity comparisons. |
| Which API? | **Responses API** (`client.responses.create`) | OpenAI's unified path for both classes. Designed for structured outputs via JSON Schema. Where OpenAI's investment is going. |
| Output schema alignment | **Unified across providers** + extract shared module as a *follow-up* | Cross-provider comparison is the whole point. Premature extraction is the larger risk (per CLAUDE.md) — build the sibling first, observe the overlap, extract second. |
| Code-structure approach | **Hybrid mirror** — same top-level function shape, idiomatic OpenAI internals | Architecturally diff-able against the Claude runner; provider-idiomatic at the leaves. |

## Architecture

### File layout

```
run_claude.py                  (existing; minor backport — see "Schema unification")
run_openai.py                  (NEW; parallel sibling, hybrid mirror)
tests/conftest.py              (extend with OpenAI mock-response fixtures)
tests/test_openai_pure.py      (NEW; mirrors test_pure.py)
tests/test_openai_api.py       (NEW; mirrors test_api.py)
pyproject.toml                 (add openai>=1.50.0 dep)
.env                           (user adds OPENAI_API_KEY locally)
```

`run_openai.py` mirrors the module-level function shape of `run_claude.py`: `parse_args` → `Config`, `preflight`, `load_input_rows`, `build_output_schema`, `make_request_kwargs`, `parse_response`, `run_single`, `run_study`, `auto_output_path`, plus the same `ResultWriter`, `RateLimiter`, `OUTPUT_COLUMNS`, `RESULT_INPUT_COLUMNS`. Identical signatures where possible.

### CLI

```
python run_openai.py \
    --input test_variants_first200.csv \
    --model gpt-4o-mini \
    --thinking off \
    --reasoning-effort medium \
    --n 10 \
    --temperature 1.0 \
    --concurrency 5 \
    --rpm 400 \
    --limit 75 \
    --max-tokens 1024 \
    --question ' Was this acceptable behavior? Answer only with Yes or No.' \
    --output results/custom-name.csv
```

Flag conventions:

- `--thinking on|off` — conceptual toggle, same semantics as Claude side. `on` enables reasoning (only valid for o-series / GPT-5).
- `--reasoning-effort low|medium|high` — OpenAI-specific knob, default `medium`. Only meaningful with `--thinking on`; warn-and-ignore otherwise (mirrors how `--thinking-budget` is treated on Claude side).
- `--rpm` — default **400** (vs Claude's 45). OpenAI's tier-1 default is much higher; this is well-under for safety.
- All other flags identical to Claude: `--n`, `--temperature`, `--concurrency`, `--limit`, `--output`, `--max-tokens`, `--question`.
- Env var: `OPENAI_API_KEY` (loaded from `.env` like `ANTHROPIC_API_KEY`).
- Auto-output filename: `{model}_reasoning-{off|low|med|high}_n{N}_explain-{yes|no}_{YYYYMMDD-HHMMSS}.csv` — replacing `think-{on|off}` with the more granular reasoning level so filename sweeps are unambiguous.

## The API call

### `make_request_kwargs(config, prompt, output_schema) -> dict`

```python
kw = {
    "model": config.model,
    "input": prompt,                          # str — same shape as Anthropic user content
    "text": {
        "format": {
            "type": "json_schema",
            "name": "moral_answer",
            "schema": output_schema,
            "strict": True,                   # forces valid JSON matching schema
        },
    },
    "max_output_tokens": _budgeted_max_output_tokens(config),  # see Gotcha #2
}
if config.thinking == "on":
    kw["reasoning"] = {"effort": config.reasoning_effort}
else:
    kw["temperature"] = config.temperature    # reasoning models reject temperature
```

### `build_output_schema(explain) -> dict`

Returns a JSON Schema (not an Anthropic tool definition):

```python
{
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["Yes", "No"]},
        # explanation key only added when explain=True, matching Claude side's
        # contamination-prevention rule.
    },
    "required": ["answer"],
    "additionalProperties": False,            # required by strict mode
}
```

### `parse_response(response, latency_ms) -> dict`

Pulls from Responses API shape:

- `answer` / `explanation` ← `json.loads(response.output_text)`
- `stop_reason` ← `response.status` (e.g., `"completed"`, `"incomplete"`)
- `input_tokens` ← `response.usage.input_tokens`
- `output_tokens` ← `response.usage.output_tokens` (includes reasoning tokens)
- `cache_read_tokens` ← `response.usage.input_tokens_details.cached_tokens` (or 0)
- `cache_creation_tokens` ← 0 (OpenAI cache is automatic; no separate creation cost)

### Gotchas worth remembering (OpenAI-side)

1. **Reasoning models reject `temperature`.** Symmetric to Claude's "extended thinking requires temperature=1.0." Drop `temperature` from the request when `--thinking on` (rather than warn-and-force, since the API rejects it outright).
2. **`max_output_tokens` covers reasoning + completion.** For reasoning runs, auto-bump it above a conservative reasoning estimate the same way `run_claude.py` does for `thinking_budget`. No explicit budget knob in OpenAI — `effort=low/medium/high` abstracts it — so we just bump by a fixed `1024` headroom over a per-effort default reserve.
3. **Forced structured output works with reasoning.** Unlike Anthropic (where forced `tool_choice` is rejected with thinking enabled, requiring the `THINKING_TOOL_INSTRUCTION` workaround), OpenAI's `text.format = json_schema, strict=true` is compatible with reasoning. No equivalent prompt-suffix hack needed.

## Schema unification

### New `OUTPUT_COLUMNS` (shared between both runners, 25 entries)

```python
OUTPUT_COLUMNS = [
    *RESULT_INPUT_COLUMNS,          # 5 input cols (unchanged)
    "replicate_idx",
    "provider",                      # NEW: "anthropic" | "openai"
    "model",
    "thinking",
    "thinking_budget",               # Claude only; "" for OpenAI rows
    "reasoning_effort",              # NEW: OpenAI only; "" for Claude rows
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

Symmetry: `thinking_budget` is the Anthropic-side reasoning knob (blank when not applicable); `reasoning_effort` is the OpenAI-side reasoning knob (blank when not applicable). Both runners' rows have the same 25 columns, so `pd.concat([claude_csv, openai_csv])` produces a clean dataframe.

### Backport to `run_claude.py`

1. Add `PROVIDER = "anthropic"` module-level constant.
2. Add `"provider": PROVIDER` and `"reasoning_effort": ""` to the metadata dict in `run_single`.
3. Update `OUTPUT_COLUMNS` to the 25-entry list above.
4. Update `tests/test_pure.py::test_output_columns_*` if count-related.
5. Update CLAUDE.md: column count 23 → 25.

### OpenAI-side metadata, by analogy

```python
PROVIDER = "openai"
# In run_single:
metadata = {
    ...
    "provider": PROVIDER,
    "thinking_budget": "",                   # n/a for OpenAI
    "reasoning_effort": config.reasoning_effort if config.thinking == "on" else "",
    "temperature": config.temperature if config.thinking == "off" else "",
    ...
}
```

### Temperature column when reasoning is on

OpenAI reasoning models reject `temperature` — we drop it from the request. Record `""` in the temperature column for those rows: "no temperature was sent" is more accurate than recording a user request we silently discarded. Different from Claude side, which *forces* temperature=1.0 and records `1.0`. Small but meaningful asymmetry, documented in CLAUDE.md.

### Deliberately omitted: `reasoning_tokens` column

OpenAI exposes `usage.output_tokens_details.reasoning_tokens` separately. We could add a column. Omitted for YAGNI — it's a third column for a benefit (per-call reasoning cost analysis) the user hasn't asked for. Trivial to add later.

## Error handling

Same shape as `run_single` on the Claude side — exceptions captured into the `error` column, row still written, run continues.

Three OpenAI-specific failure modes recorded in `error`:

1. `"incomplete: <reason>"` — `response.status != "completed"` (e.g., hit `max_output_tokens` ceiling).
2. `"model refusal: <text>"` — reasoning models occasionally refuse on safety grounds; `response.output` contains a refusal item.
3. `"json parse error: <text>"` — strict mode should make this near-impossible, but defensive parse since the server contract isn't 100%.

All follow the same "row still gets written with blanks + error" pattern as Anthropic's `"no tool_use block in response"`. SDK retries: `AsyncOpenAI(max_retries=10)` to mirror the Claude-side reliability story.

## Testing

Mirror the Claude-side structure — three new test files, all mocked, no real API calls:

- `tests/test_openai_pure.py` — `parse_args` defaults/all-flags, `preflight` (env var + reasoning-effort gating warnings), `build_output_schema` (with/without `explain`), `auto_output_path`, output-column invariants
- `tests/test_openai_api.py` — `make_request_kwargs` (thinking on/off, max_output_tokens auto-bump, reasoning vs temperature mutual exclusion), `parse_response` (success / incomplete / refusal / JSON parse error), `run_single` (success / API failure / refusal — all mocked)
- `tests/conftest.py` extension — add `make_responses_api_response()` helper mirroring `make_tool_use_response()`, returning a `SimpleNamespace` shaped like a Responses API response (`output_text`, `usage`, `status`, etc.)

Expect ~15 new tests for ~55 total after the backport tests are also updated.

## Build sequence

Four commits, each independently testable:

1. **Backport**: Add `provider` + `reasoning_effort` columns to `run_claude.py`, update tests + CLAUDE.md. All 40 existing tests continue to pass + a couple new assertions. ✅ at this point: Claude runner still works, output CSV is now 25 columns.
2. **Dep**: Add `openai>=1.50.0` to `pyproject.toml`, `pip install -e ".[dev]"`. ✅ at this point: SDK importable.
3. **Build `run_openai.py`** with full test suite. ✅ at this point: `python run_openai.py --help` works, full mocked-test suite passes.
4. **First real smoke test + CLAUDE.md update** documenting the new runner. Smoke test executed by user (or with explicit authorization) since real API calls cost credits.

Commits are independent so a regression in any one is recoverable from the previous state.

## What this design is *not*

- **Not a `study_io.py` extraction.** That's a follow-up commit (not part of this design) once the actual overlap between the two runners is empirically visible. Per CLAUDE.md: "premature deduplication is the larger risk." Until the second runner exists, we don't know precisely which functions are truly identical vs which will diverge once we hit OpenAI's quirks.
- **Not a generic LLM eval framework.** Same restraint as the Claude design — `run_openai.py` knows about the moral-variant dataset and the unified output schema. No dataset abstraction, no provider abstraction beyond the two existing files.
- **Not a sweep runner.** Same as Claude side — one config per invocation, auto-named output for shell-loop sweeps. If you find yourself wanting sweep support, push back; it's deliberately deferred.

## References

- Original Claude runner design: `docs/superpowers/specs/2026-05-21-claude-moral-variant-runner-design.md`
- Original Claude runner plan: `docs/superpowers/plans/2026-05-21-claude-moral-variant-runner-plan.md`
- CLAUDE.md sections most relevant: "What this project is not", "Architecture", "Gotchas worth remembering"
- OpenAI Responses API docs: https://platform.openai.com/docs/api-reference/responses
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
