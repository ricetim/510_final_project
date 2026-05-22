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
