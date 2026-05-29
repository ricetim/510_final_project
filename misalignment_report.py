#!/usr/bin/env python3
"""Per-group divergence from consensus on contested scenarios.

For each contested scenario (one where the 15 demographic variants don't
all agree on Yes vs No), we measure how far each group's yes-rate sits
from the consensus of the OTHER voters — leave-one-out at every group
level.

Two metrics per group:
  - mean |group_yr - others_yr|:  how far the group sits from consensus
    on average. 0 = perfectly tracks consensus. Higher = consistently
    out of step (regardless of direction).
  - mean (group_yr - others_yr):  signed bias. Positive = group is more
    permissive (more Yes) than the rest; negative = group is stricter
    (more No) than the rest.

Group definitions:
  - Per-variant (15):  the variant's own 5 replicates vs the other 70.
  - Per-race (5):      the race's 15 replicates vs the other 60.
  - Per-income (3):    the income's 25 replicates vs the other 50.

Why leave-one-out: if you put a group's own replicates into its own
consensus, you create a self-pull bias whose size depends on group
size — a 25-rep income tier drags the "consensus" toward its own
position by ~33%, while a 5-rep variant only drags it ~7%. Across-level
comparisons (per-variant vs per-income) would then be apples-to-oranges.
Computing the consensus over everyone-but-this-group removes that.

A scenario is contested if the 15 variants don't unanimously agree on Yes
vs No. ~80% of scenarios on Haiku think-off are unanimous and get
dropped; including them would dilute every group's distance toward 0.

Usage:
    python misalignment_report.py <results_csv> [--output report.html]
                                                [--include-unanimous]
"""

import argparse
import csv
import html as _html
import sys
from collections import defaultdict
from pathlib import Path

from variant_trends_report import (
    RACES, INCOMES, VARIANTS,
    load_yes_rate_matrix, is_unanimous,
    _sanitize_for_filename, resolve_non_clobbering,
)


def esc(value) -> str:
    return _html.escape(str(value) if value is not None else "")


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def collect_counts(raw_rows: list[dict], contested_set: set[str]):
    """Tally yes-and-total counts per scenario at three group levels.

    Returns:
        global_counts:  {s: [n_yes, n_total]}
        variant_counts: {s: {(race, income): [n_yes, n_total]}}
        race_counts:    {s: {race: [n_yes, n_total]}}
        income_counts:  {s: {income: [n_yes, n_total]}}
    """
    global_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    variant_counts: dict[str, dict[tuple, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0]))
    race_counts: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0]))
    income_counts: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0]))

    for r in raw_rows:
        s = r["scenario_id"]
        if s not in contested_set:
            continue
        if r["answer"] not in ("Yes", "No"):
            continue
        race = r["race_variant"]
        inc = r["income_variant"]
        if race not in RACES or inc not in INCOMES:
            continue
        y = 1 if r["answer"] == "Yes" else 0

        global_counts[s][0] += y
        global_counts[s][1] += 1
        variant_counts[s][(race, inc)][0] += y
        variant_counts[s][(race, inc)][1] += 1
        race_counts[s][race][0] += y
        race_counts[s][race][1] += 1
        income_counts[s][inc][0] += y
        income_counts[s][inc][1] += 1

    return global_counts, variant_counts, race_counts, income_counts


def leave_one_out_divergence(group_counts_per_scenario: dict,
                             global_counts: dict,
                             groups: list,
                             contested_set: set[str]
                             ) -> dict:
    """Per group: mean |group_yr - others_yr| and mean signed (group - others).

    others = all replicates in the scenario MINUS those in this group.

    Returns {group: (mean_abs, mean_signed, n_scenarios)}.
    """
    out: dict = {}
    for g in groups:
        abs_dists: list[float] = []
        signed_diffs: list[float] = []
        for s in contested_set:
            gpair = global_counts.get(s)
            cell = group_counts_per_scenario.get(s, {}).get(g)
            if not gpair or not cell:
                continue
            global_yes, global_total = gpair
            group_yes, group_total = cell
            others_yes = global_yes - group_yes
            others_total = global_total - group_total
            if group_total == 0 or others_total == 0:
                continue
            group_yr = group_yes / group_total
            others_yr = others_yes / others_total
            abs_dists.append(abs(group_yr - others_yr))
            signed_diffs.append(group_yr - others_yr)
        if not abs_dists:
            out[g] = (float("nan"), float("nan"), 0)
        else:
            out[g] = (sum(abs_dists) / len(abs_dists),
                      sum(signed_diffs) / len(signed_diffs),
                      len(abs_dists))
    return out


# ---------------------------------------------------------------------------
# Stdout output
# ---------------------------------------------------------------------------

def print_table(title: str, rows: list[tuple[str, float, float, int]]) -> None:
    print(f"\n=== {title} ===")
    print(f"  {'group':<28}{'|dist|':>10}{'signed':>10}{'  n':>6}")
    print(f"  {'-' * 28}{'-' * 10}{'-' * 10}{'-' * 6}")
    keyed = sorted(rows, key=lambda r: -(r[1] if r[1] == r[1] else -1))
    for label, mabs, msign, n in keyed:
        mabs_s = "—" if mabs != mabs else f"{mabs:.3f}"
        msign_s = "—" if msign != msign else f"{msign:+.3f}"
        print(f"  {label:<28}{mabs_s:>10}{msign_s:>10}{n:>6}")


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def render_bar(rate: float) -> str:
    """Sequential blue bar; saturates at 0.5 (typical practical max)."""
    if rate != rate:
        return '<div class="bar-cell">&mdash;</div>'
    s = max(0.0, min(1.0, rate / 0.5))
    pct = s * 100
    light = 100 - 55 * s
    return (f'<div class="bar-cell">'
            f'<div class="bar" style="width:{pct:.0f}%;'
            f'background:hsl(220, 60%, {light:.0f}%);"></div>'
            f'<span class="bar-label">{rate:.3f}</span>'
            f'</div>')


def render_signed_bar(rate: float) -> str:
    """Diverging bar: center at 50%; positive blue right, negative red left."""
    if rate != rate:
        return '<div class="bar-cell signed">&mdash;</div>'
    s = max(-1.0, min(1.0, rate / 0.5))
    half = abs(s) * 50  # half-width in percent
    light = 100 - 55 * abs(s)
    if rate >= 0:
        bar = (f'<div class="bar" style="left:50%;width:{half:.1f}%;'
               f'background:hsl(220, 60%, {light:.0f}%);"></div>')
    else:
        bar = (f'<div class="bar" style="right:50%;width:{half:.1f}%;'
               f'background:hsl(0, 60%, {light:.0f}%);"></div>')
    sign = "+" if rate >= 0 else ""
    return (f'<div class="bar-cell signed">'
            f'<div class="midline"></div>{bar}'
            f'<span class="bar-label">{sign}{rate:.3f}</span></div>')


def render_html_table(rows: list[tuple[str, float, float, int]]) -> str:
    parts = ['<table class="rates"><thead><tr>'
             '<th>group</th>'
             '<th>mean |group &minus; others| (distance from consensus)</th>'
             '<th>mean (group &minus; others) &mdash; signed bias</th>'
             '<th class="numhead">n</th>'
             '</tr></thead><tbody>']
    keyed = sorted(rows, key=lambda r: -(r[1] if r[1] == r[1] else -1))
    for label, mabs, msign, n in keyed:
        parts.append(
            f'<tr><th>{esc(label)}</th>'
            f'<td>{render_bar(mabs)}</td>'
            f'<td>{render_signed_bar(msign)}</td>'
            f'<td class="num">{n}</td></tr>'
        )
    parts.append("</tbody></table>")
    return "".join(parts)


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1200px; margin: 2em auto; padding: 0 1em; color: #222; }
h1 { font-size: 1.55em; margin-bottom: 0.2em; }
h2 { font-size: 1.2em; margin-top: 2.2em; border-bottom: 2px solid #333;
     padding-bottom: 0.3em; }
.meta { color: #666; font-size: 0.9em; margin-bottom: 1.5em; }
.note { color: #555; font-size: 0.92em; max-width: 86ch; margin: 0.5em 0 1em 0; }
.note p { margin: 0.5em 0; }
.note ul { margin: 0.3em 0 0.6em 0; padding-left: 1.5em; }
.note li { margin: 0.15em 0; }
.note em { color: #333; font-style: normal; font-weight: 500; }
table.rates { border-collapse: collapse; width: 100%; max-width: 1040px;
              font-size: 0.92em; margin: 0.5em 0 1em 0; }
table.rates th, table.rates td { border: 1px solid #ddd; padding: 4px 8px;
                                  vertical-align: middle; }
table.rates thead th { background: #efefef; text-align: left; font-weight: 600; }
table.rates thead th.numhead { text-align: right; width: 6%; }
table.rates tbody th { text-align: left; font-weight: 500; background: #f7f7f7;
                       width: 22%; }
table.rates td.num { text-align: right; font-variant-numeric: tabular-nums;
                     width: 6%; }
.bar-cell { position: relative; height: 22px; min-width: 280px; background: #fafafa; }
.bar-cell.signed { background: #f5f5f5; }
.bar-cell .midline { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px;
                     background: #888; z-index: 1; }
.bar { position: absolute; top: 0; bottom: 0; }
.bar-label { position: absolute; left: 50%; top: 3px; transform: translateX(-50%);
             font-variant-numeric: tabular-nums; font-size: 0.85em; color: #222;
             text-shadow: 0 0 3px white, 0 0 2px white; z-index: 2; }
.bar-cell:not(.signed) .bar-label { left: 8px; transform: none; }
"""


def build_html(results_path: Path, model: str,
               n_total: int, n_contested: int,
               per_var: dict, per_race: dict, per_income: dict) -> str:
    var_rows = [(f"{r} / {i.replace(' income', '')}", *per_var[(r, i)])
                for r, i in VARIANTS]
    race_rows = [(r, *per_race[r]) for r in RACES]
    income_rows = [(i.replace(" income", ""), *per_income[i]) for i in INCOMES]

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Per-group divergence from consensus</title>
<style>{CSS}</style></head><body>
<h1>Per-group divergence from consensus &mdash; contested scenarios</h1>
<div class="meta">Source: <code>{esc(results_path)}</code> &middot;
Model: <code>{esc(model)}</code> &middot;
{n_contested} contested of {n_total} full-coverage scenarios</div>

<div class="note">
<p><strong>Method:</strong></p>
<ul>
<li>A scenario is <em>contested</em> if its 15 demographic variants
don&rsquo;t unanimously agree on Yes vs No. (~80% of scenarios are
unanimous and get filtered out &mdash; they carry no signal.)</li>
<li>For each group we compute the group&rsquo;s yes-rate over its own
replicates, and compare against the <em>leave-one-out consensus</em>:
the yes-rate over every replicate in the scenario <strong>not</strong>
in this group. This is the &ldquo;others' opinion of this scenario&rdquo;
that the group is being measured against.</li>
<li>Two metrics per group:
<ul>
<li><em>mean &vert;group &minus; others&vert;</em>: how far the group
sits from the others&rsquo; consensus on average. 0 = always tracks
consensus; higher = consistently out of step (in either direction). The
bar saturates at 0.5.</li>
<li><em>mean (group &minus; others)</em>: signed bias. Positive (blue)
= group is more permissive (more Yes) than the rest. Negative (red) =
group is stricter (more No).</li>
</ul></li>
<li><em>n</em> = number of contested scenarios that contributed to the
group&rsquo;s averages.</li>
</ul>
<p><strong>Reading the two columns together:</strong> A large
&vert;distance&vert; with a near-zero signed bias means the group
disagrees with consensus often but in <em>both</em> directions
(sometimes more permissive, sometimes stricter). A large
&vert;distance&vert; with a strong same-sign bias means the group
reliably leans one way.</p>
</div>

<h2>Per income tier (3 groups, pooling across race)</h2>
{render_html_table(income_rows)}

<h2>Per race (5 groups, pooling across income)</h2>
{render_html_table(race_rows)}

<h2>Per (race, income) variant (15 groups)</h2>
{render_html_table(var_rows)}

</body></html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("results_csv", help="Path to results CSV.")
    p.add_argument("--output", default=None,
                   help="Output HTML path. Default: "
                        "reports/misalignment_<model>.html.")
    p.add_argument("--include-unanimous", action="store_true",
                   help="Keep unanimous scenarios (every group has "
                        "distance 0 on them, so they only dilute the "
                        "averages; off by default).")
    args = p.parse_args()

    results_path = Path(args.results_csv)
    if not results_path.exists():
        print(f"error: results CSV not found: {results_path}", file=sys.stderr)
        sys.exit(2)

    # Use the per-variant 5-rep majority logic to define "contested".
    scenarios, rates, variant_keys = load_yes_rate_matrix(results_path, axis="both")
    n_total = len(scenarios)

    if args.include_unanimous:
        contested = scenarios
    else:
        contested = [s for s in scenarios if not is_unanimous(rates, s, variant_keys)]
    contested_set = set(contested)
    n_contested = len(contested)

    if not n_contested:
        print("error: no contested scenarios after filtering.", file=sys.stderr)
        sys.exit(2)

    # Re-read raw rows for replicate-level counting.
    with results_path.open(newline="") as f:
        raw_rows = list(csv.DictReader(f))

    global_counts, variant_counts, race_counts, income_counts = \
        collect_counts(raw_rows, contested_set)

    per_var = leave_one_out_divergence(variant_counts, global_counts,
                                       VARIANTS, contested_set)
    per_race = leave_one_out_divergence(race_counts, global_counts,
                                        RACES, contested_set)
    per_income = leave_one_out_divergence(income_counts, global_counts,
                                          INCOMES, contested_set)

    model = next((r.get("model", "") for r in raw_rows if r.get("model")), "")

    # Stdout summary.
    print(f"Source: {results_path}")
    print(f"Contested scenarios: {n_contested} of {n_total} full-coverage")

    print_table(
        "Per income (pooled across race)",
        [(i.replace(" income", ""), *per_income[i]) for i in INCOMES],
    )
    print_table(
        "Per race (pooled across income)",
        [(r, *per_race[r]) for r in RACES],
    )
    print_table(
        "Per (race, income) variant",
        [(f"{r} / {i.replace(' income', '')}", *per_var[(r, i)])
         for r, i in VARIANTS],
    )

    # HTML
    requested = Path(args.output) if args.output else (
        Path(__file__).resolve().parent / "reports"
        / f"misalignment_{_sanitize_for_filename(model)}.html"
    )
    output_path = resolve_non_clobbering(requested)
    if output_path != requested:
        print(f"\nwarning: {requested} exists; writing to {output_path} instead",
              file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_html(results_path, model, n_total, n_contested,
                   per_var, per_race, per_income),
        encoding="utf-8",
    )
    print(f"\nHTML report: {output_path}")


if __name__ == "__main__":
    main()
