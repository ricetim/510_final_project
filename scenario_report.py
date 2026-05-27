#!/usr/bin/env python3
"""Generate an HTML report of all results for given scenario_ids from a results CSV.

Shows every row for each scenario, grouped by (race, income) variant — including
rows with parse errors, blank answers, or refusals. Nothing is filtered out;
the point is to see model misbehavior alongside successes.

Usage:
    python scenario_report.py <results_csv> <scenario_id> [<scenario_id> ...] [--output report.html]
    python scenario_report.py <results_csv> --scenario-file scenarios.txt [--output report.html]
    python scenario_report.py <results_csv> 00001 00051 --scenario-file more.txt

Notes:
    - scenario_id matching is exact-string. The dataset uses 5-digit zero-padded
      IDs like "00051"; "51" will not match.
    - --scenario-file is one ID per line, blanks ignored.
    - Default output path: scenario_report_<N>.html in the CWD.
"""

import argparse
import csv
import html as _html
import sys
from collections import defaultdict
from pathlib import Path


def esc(value) -> str:
    return _html.escape(str(value) if value is not None else "")


def build_html(by_scenario: dict, requested: list[str], results_path: Path) -> str:
    parts: list[str] = []
    parts.append(
        """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Scenario report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 1200px; margin: 2em auto; padding: 0 1em; color: #222; }
  h1 { font-size: 1.6em; margin-bottom: 0.2em; }
  h2 { font-size: 1.3em; margin-top: 2em; border-bottom: 2px solid #333; padding-bottom: 0.3em; }
  h3 { font-size: 1.05em; margin-top: 1em; }
  .meta { color: #666; font-size: 0.9em; margin-bottom: 1.5em; }
  .missing { color: #b13030; }
  .scenario { margin-bottom: 3em; }
  .original { background: #f7f7f7; padding: 0.75em 1em; border-left: 4px solid #888;
              margin: 0.5em 0 1em 0; font-size: 0.95em; }
  .summary { font-size: 0.9em; color: #444; margin-bottom: 0.5em; }
  .variant { background: #fafafd; padding: 0.75em 1em; margin: 0.75em 0; border-left: 3px solid #6cf; }
  .prompt { color: #555; font-size: 0.9em; margin: 0.25em 0 0.75em 0; }
  table { border-collapse: collapse; width: 100%; font-size: 0.85em; margin-top: 0.5em; }
  th, td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; vertical-align: top; }
  th { background: #efefef; }
  td.answer-yes { color: #1a8830; font-weight: 600; }
  td.answer-no { color: #b13030; font-weight: 600; }
  td.answer-blank { color: #999; font-style: italic; }
  td.error-cell { color: #b13030; font-size: 0.85em; max-width: 30em; }
  td.numeric { text-align: right; font-variant-numeric: tabular-nums; }
  td.explanation { max-width: 32em; }
</style>
</head>
<body>
"""
    )
    parts.append(f"<h1>Scenario report</h1>")
    parts.append(
        f'<div class="meta">Source: <code>{esc(results_path)}</code> · '
        f"Requested: {esc(', '.join(requested))} · "
        f"Found: {len(by_scenario)}/{len(requested)}</div>"
    )

    missing = sorted(set(requested) - set(by_scenario.keys()))
    if missing:
        parts.append(
            f'<div class="meta missing">Not found in results: {esc(", ".join(missing))}</div>'
        )

    for sid in sorted(by_scenario.keys()):
        variants = by_scenario[sid]
        sample = next(iter(variants.values()))[0]
        original = sample.get("original_scenario", "")
        all_rows = [r for v in variants.values() for r in v]
        n_total = len(all_rows)
        n_yes = sum(1 for r in all_rows if r["answer"] == "Yes")
        n_no = sum(1 for r in all_rows if r["answer"] == "No")
        n_err = sum(1 for r in all_rows if r.get("error"))
        n_blank = sum(
            1 for r in all_rows if r["answer"] not in ("Yes", "No") and not r.get("error")
        )

        parts.append('<div class="scenario">')
        parts.append(f"<h2>Scenario {esc(sid)}</h2>")
        parts.append(
            f'<div class="original"><strong>Original:</strong> {esc(original)}</div>'
        )
        parts.append(
            f'<div class="summary">{n_total} total · '
            f"{n_yes} Yes · {n_no} No · {n_err} errors · {n_blank} blank · "
            f"{len(variants)} variant combos</div>"
        )

        for combo in sorted(variants.keys()):
            v_rows = variants[combo]
            v_sample = v_rows[0]
            v_yes = sum(1 for r in v_rows if r["answer"] == "Yes")
            v_err = sum(1 for r in v_rows if r.get("error"))
            yes_rate = v_yes / len(v_rows) if v_rows else 0.0
            race, income = combo

            parts.append('<div class="variant">')
            parts.append(
                f"<h3>{esc(race)} / {esc(income)} — yes-rate "
                f"{yes_rate:.2f} ({v_yes}/{len(v_rows)}) · {v_err} errors</h3>"
            )
            full_prompt = v_sample.get("variant_scenario", "") + v_sample.get("question", "")
            parts.append(
                f'<div class="prompt"><strong>Prompt sent:</strong> {esc(full_prompt)}</div>'
            )

            parts.append("<table>")
            parts.append(
                "<thead><tr>"
                "<th>#</th><th>Answer</th><th>Explanation</th><th>Error</th>"
                "<th>Stop</th><th>In tok</th><th>Out tok</th><th>Lat (ms)</th>"
                "</tr></thead><tbody>"
            )
            for r in sorted(v_rows, key=lambda x: int(x.get("replicate_idx", 0))):
                ans = r["answer"]
                if ans == "Yes":
                    cls = "answer-yes"
                    shown = "Yes"
                elif ans == "No":
                    cls = "answer-no"
                    shown = "No"
                else:
                    cls = "answer-blank"
                    shown = "(blank)"
                err_cls = "error-cell" if r.get("error") else ""
                parts.append(
                    "<tr>"
                    f'<td class="numeric">{esc(r.get("replicate_idx", ""))}</td>'
                    f'<td class="{cls}">{esc(shown)}</td>'
                    f'<td class="explanation">{esc(r.get("explanation", ""))}</td>'
                    f'<td class="{err_cls}">{esc(r.get("error", ""))}</td>'
                    f'<td>{esc(r.get("stop_reason", ""))}</td>'
                    f'<td class="numeric">{esc(r.get("input_tokens", ""))}</td>'
                    f'<td class="numeric">{esc(r.get("output_tokens", ""))}</td>'
                    f'<td class="numeric">{esc(r.get("latency_ms", ""))}</td>'
                    "</tr>"
                )
            parts.append("</tbody></table>")
            parts.append("</div>")

        parts.append("</div>")

    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("results_csv", help="Path to results CSV.")
    p.add_argument("scenarios", nargs="*", help="Scenario ID(s), e.g. 00001 00051.")
    p.add_argument(
        "--scenario-file",
        default=None,
        help="Text file with scenario IDs, one per line.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output HTML path. Default: scenario_report_<N>.html.",
    )
    args = p.parse_args()

    scenarios = list(args.scenarios)
    if args.scenario_file:
        with open(args.scenario_file) as f:
            for line in f:
                sid = line.strip()
                if sid:
                    scenarios.append(sid)

    if not scenarios:
        print(
            "error: provide at least one scenario id (positional or via --scenario-file).",
            file=sys.stderr,
        )
        sys.exit(2)

    scenarios = sorted(set(scenarios))

    results_path = Path(args.results_csv)
    if not results_path.exists():
        print(f"error: results CSV not found: {results_path}", file=sys.stderr)
        sys.exit(2)

    requested_set = set(scenarios)
    with results_path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["scenario_id"] in requested_set]

    if not rows:
        print(
            f"error: no rows found for any of: {', '.join(scenarios)}",
            file=sys.stderr,
        )
        sys.exit(2)

    by_scenario: dict[str, dict[tuple[str, str], list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        by_scenario[r["scenario_id"]][(r["race_variant"], r["income_variant"])].append(r)

    output_path = (
        Path(args.output)
        if args.output
        else Path(f"scenario_report_{len(scenarios)}.html")
    )
    output_path.write_text(build_html(by_scenario, scenarios, results_path), encoding="utf-8")

    n_rows = sum(len(v) for s in by_scenario.values() for v in s.values())
    print(
        f"wrote {n_rows} rows across {len(by_scenario)} scenarios to {output_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
