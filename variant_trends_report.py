#!/usr/bin/env python3
"""Find which demographic variants answer together (and which oppose).

Takes a results CSV and emits a single HTML report with two views:

  1. Pairwise agreement matrix — NxN heatmap of "fraction of scenarios on
     which these two variants gave the same majority answer." Diagonal = 1.
  2. Spearman correlation matrix — NxN heatmap of rank correlation on
     per-scenario yes-rates. Diverging color scale: blue = answer together,
     red = answer opposite. Captures partial agreement that #1 misses.

Usage:
    python variant_trends_report.py <results_csv> [--output report.html]
                                                  [--include-unanimous]
                                                  [--axis both|race|income]

--axis controls how variants are formed:
  - both   (default): 15 crossed (race, income) variants — original view.
  - race:   collapse to 5 race categories, pooling all 3 income tiers per
            race. Each scenario's yes-rate for "white" is computed from
            #Yes / (#Yes + #No) across all 15 replicates (5 reps × 3
            income tiers), so the underlying-replicate noise model is
            preserved (no double-averaging).
  - income: collapse to 3 income categories, pooling all 5 races per
            income tier. Same pooling logic, 25 replicates per cell.

Scenarios where all N variants give the SAME majority answer (everyone
Yes, or everyone No) carry no demographic signal — they pull every pair's
agreement / correlation toward each other without telling you anything
about bias. By default they're dropped before the matrices are built;
pass --include-unanimous to keep them.

Default output: reports/variant_trends_<model>[_axis-<race|income>].html,
anchored to the script's parent dir (not CWD), no-clobber.

No numpy/scipy required. Everything is implemented in pure Python because
N <= 15 variants is small enough that O(n^2) and O(n^3) routines run in
microseconds.
"""

import argparse
import csv
import html as _html
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

# Canonical variant ordering. Keeping race outermost makes the matrix
# group-by-race in reading order, which is what you'd want when scanning
# for "do all the income tiers within one race cluster?"
RACES = ["white", "black", "latino", "east asian", "indigenous"]
INCOMES = ["low income", "middle income", "high income"]
VARIANTS: list[tuple[str, str]] = [(r, i) for r in RACES for i in INCOMES]
VARIANT_LABELS = [f"{r} / {i.replace(' income', '')}" for r, i in VARIANTS]


def axis_categories(axis: str) -> tuple[list, list[str]]:
    """Return (category_keys, display_labels) for the given axis.

    - axis="both"   -> 15 (race, income) tuples + the cross-product labels.
    - axis="race"   -> 5 race strings + the same as labels.
    - axis="income" -> 3 income strings + labels with " income" stripped.
    """
    if axis == "race":
        return list(RACES), list(RACES)
    if axis == "income":
        return list(INCOMES), [i.replace(" income", "") for i in INCOMES]
    return list(VARIANTS), list(VARIANT_LABELS)


def esc(value) -> str:
    return _html.escape(str(value) if value is not None else "")


def _sanitize_for_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value) or "unknown"


def default_output_path(model: str, axis: str = "both") -> Path:
    suffix = "" if axis == "both" else f"_axis-{axis}"
    return Path(__file__).resolve().parent / "reports" \
        / f"variant_trends_{_sanitize_for_filename(model)}{suffix}.html"


def resolve_non_clobbering(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    i = 2
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------

def load_yes_rate_matrix(path: Path, axis: str = "both",
                         require_complete: bool = True
                         ) -> tuple[list[str], dict, list]:
    """Return (scenario_ids_sorted, variant -> {scenario_id: yes_rate}, variant_keys).

    With require_complete=True (default), only scenarios where every variant
    has at least one Yes/No reply are kept. With require_complete=False, any
    scenario with at least one cell populated is kept and callers must
    tolerate `sid not in rates[v]` for the missing cells (the matrix
    builders here do — they fall back to pairwise complete observations).

    Pooling note: for the collapsed axes the yes-rate for one cell on one
    scenario is computed from ALL underlying replicates (5 reps * 3 incomes
    = 15 for axis=race; 5 reps * 5 races = 25 for axis=income). This avoids
    double-averaging — if you averaged the three already-summarized
    yes-rates instead, scenarios where recovery yielded different replicate
    counts per cell would get silently re-weighted.
    """
    variant_keys, _ = axis_categories(axis)
    if axis == "race":
        keyfn = lambda r: r["race_variant"]
    elif axis == "income":
        keyfn = lambda r: r["income_variant"]
    else:
        keyfn = lambda r: (r["race_variant"], r["income_variant"])
    valid = set(variant_keys)

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    # variant -> scenario -> [answers]
    raw: dict = defaultdict(lambda: defaultdict(list))
    all_scenarios: set[str] = set()
    for r in rows:
        if r["answer"] not in ("Yes", "No"):
            continue
        # Guard: drop any row whose race/income isn't in the canonical lists
        # so a stray demographic typo in the CSV can't mask a missing cell.
        if r["race_variant"] not in RACES or r["income_variant"] not in INCOMES:
            continue
        k = keyfn(r)
        if k not in valid:
            continue
        raw[k][r["scenario_id"]].append(r["answer"])
        all_scenarios.add(r["scenario_id"])

    # Compute yes-rates
    rates: dict = {v: {} for v in variant_keys}
    for v in variant_keys:
        for sid, answers in raw[v].items():
            rates[v][sid] = sum(1 for a in answers if a == "Yes") / len(answers)

    if require_complete:
        kept = sorted(
            sid for sid in all_scenarios
            if all(sid in rates[v] for v in variant_keys)
        )
        rates = {v: {sid: rates[v][sid] for sid in kept} for v in variant_keys}
    else:
        kept = sorted(all_scenarios)
    return kept, rates, variant_keys


# ---------------------------------------------------------------------------
# Method 1: pairwise agreement on majority votes
# ---------------------------------------------------------------------------

def majority(yes_rate: float) -> str | None:
    if yes_rate > 0.5:
        return "Yes"
    if yes_rate < 0.5:
        return "No"
    return None  # 0.5 = ambiguous; exclude from this method


def is_unanimous(rates: dict, sid: str, variant_keys: list) -> bool:
    """True if every variant that has data on this scenario agrees on the
    same Yes/No majority. Variants without data are skipped (the scenario
    can still be unanimous over its covered subset). A variant with an
    exactly-50/50 yes-rate (no majority) prevents unanimity; a scenario
    with zero covered variants returns False (no signal to count)."""
    seen: set[str] = set()
    for v in variant_keys:
        if sid not in rates[v]:
            continue
        m = majority(rates[v][sid])
        if m is None:
            return False
        seen.add(m)
    if not seen:
        return False
    return len(seen) == 1


def agreement_matrix(scenarios: list[str],
                     rates: dict,
                     variant_keys: list) -> list[list[float]]:
    """Pairwise complete observations: each pair's count is built from
    scenarios where BOTH variants have data and both have a defined Yes/No
    majority (yes-rate != 0.5)."""
    n = len(variant_keys)
    out = [[0.0] * n for _ in range(n)]
    for i, vi in enumerate(variant_keys):
        for j, vj in enumerate(variant_keys):
            if i == j:
                out[i][j] = 1.0
                continue
            agree = total = 0
            for sid in scenarios:
                if sid not in rates[vi] or sid not in rates[vj]:
                    continue
                mi = majority(rates[vi][sid])
                mj = majority(rates[vj][sid])
                if mi is None or mj is None:
                    continue
                total += 1
                if mi == mj:
                    agree += 1
            out[i][j] = (agree / total) if total else float("nan")
    return out


# ---------------------------------------------------------------------------
# Method 2: Spearman rank correlation
# ---------------------------------------------------------------------------

def midranks(xs: list[float]) -> list[float]:
    """1-indexed mid-ranks (ties averaged) — matches scipy.stats.rankdata."""
    idx = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # midpoint of 1-indexed rank range
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0  # zero variance → undefined; treat as no correlation
    return num / (dx * dy)


def spearman_matrix(scenarios: list[str],
                    rates: dict,
                    variant_keys: list) -> list[list[float]]:
    """Pairwise complete observations: for each (i, j) pair we restrict to
    scenarios where BOTH variants have data, rank within that subset, then
    Pearson-correlate. Each pair can use a different scenario count."""
    n = len(variant_keys)
    out = [[0.0] * n for _ in range(n)]
    for i, vi in enumerate(variant_keys):
        for j, vj in enumerate(variant_keys):
            if i == j:
                out[i][j] = 1.0
                continue
            xs: list[float] = []
            ys: list[float] = []
            for sid in scenarios:
                if sid not in rates[vi] or sid not in rates[vj]:
                    continue
                xs.append(rates[vi][sid])
                ys.append(rates[vj][sid])
            if len(xs) < 2:
                out[i][j] = float("nan")
                continue
            out[i][j] = pearson(midranks(xs), midranks(ys))
    return out


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def hsl_for_agreement(v: float) -> str:
    """White at 0, dark blue at 1. NaN → light gray."""
    if math.isnan(v):
        return "#eee"
    light = 100 - 55 * max(0.0, min(1.0, v))
    return f"hsl(220, 60%, {light:.0f}%)"


def hsl_for_corr(v: float) -> str:
    """Diverging: red at -1, white at 0, blue at +1. NaN → light gray."""
    if math.isnan(v):
        return "#eee"
    v = max(-1.0, min(1.0, v))
    if v >= 0:
        return f"hsl(220, 60%, {100 - 55 * v:.0f}%)"
    return f"hsl(0, 60%, {100 - 55 * abs(v):.0f}%)"


def row_means_excluding_self(matrix: list[list[float]]) -> list[float]:
    """Mean of each row excluding the diagonal cell (self-similarity)."""
    n = len(matrix)
    out: list[float] = []
    for i in range(n):
        vals = [matrix[i][j] for j in range(n) if j != i and not math.isnan(matrix[i][j])]
        out.append(sum(vals) / len(vals) if vals else float("nan"))
    return out


def render_matrix(matrix: list[list[float]],
                  labels: list[str],
                  color_fn,
                  fmt: str = "{:.2f}") -> str:
    means = row_means_excluding_self(matrix)
    parts: list[str] = ['<table class="matrix"><thead><tr><th></th>']
    for lbl in labels:
        parts.append(f'<th class="rot"><div><span>{esc(lbl)}</span></div></th>')
    parts.append('<th class="rot rowmean-head"><div><span>row mean (excl. self)</span></div></th>')
    parts.append("</tr></thead><tbody>")
    for i, lbl in enumerate(labels):
        parts.append(f'<tr><th class="row">{esc(lbl)}</th>')
        for j in range(len(labels)):
            v = matrix[i][j]
            cell = "—" if math.isnan(v) else fmt.format(v)
            parts.append(
                f'<td style="background:{color_fn(v)};">{esc(cell)}</td>'
            )
        m = means[i]
        m_cell = "—" if math.isnan(m) else fmt.format(m)
        parts.append(
            f'<td class="rowmean" style="background:{color_fn(m)};">{esc(m_cell)}</td>'
        )
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1400px; margin: 2em auto; padding: 0 1em; color: #222; }
h1 { font-size: 1.6em; margin-bottom: 0.2em; }
h2 { font-size: 1.25em; margin-top: 2.5em; border-bottom: 2px solid #333;
     padding-bottom: 0.3em; }
.meta { color: #666; font-size: 0.9em; margin-bottom: 1.5em; }
.note { color: #555; font-size: 0.9em; max-width: 80ch; margin: 0.5em 0 1em 0; }
.note p { margin: 0.5em 0; }
.note ul { margin: 0.3em 0 0.6em 0; padding-left: 1.5em; }
.note li { margin: 0.15em 0; }
.note em { color: #333; font-style: normal; font-weight: 500; }
table.matrix { border-collapse: collapse; margin: 0.5em 0; font-size: 0.78em; }
table.matrix th, table.matrix td { border: 1px solid #ccc; padding: 4px 6px;
                                   text-align: center; font-variant-numeric: tabular-nums; }
table.matrix th.row { text-align: right; font-weight: 500; background: #f3f3f3; }
table.matrix th.rot { vertical-align: bottom; height: 110px; padding: 0; }
table.matrix th.rot > div { transform: rotate(-55deg); transform-origin: bottom left;
                            width: 20px; margin-left: 6px; white-space: nowrap; }
table.matrix th.rot > div > span { padding: 2px 4px; }
table.matrix th.rowmean-head, table.matrix td.rowmean {
    border-left: 2px solid #444; font-weight: 600;
}
.legend { display: inline-flex; gap: 1.2em; align-items: center; font-size: 0.85em;
          color: #555; margin: 0.5em 0; }
.legend .swatch { display: inline-block; width: 18px; height: 12px; vertical-align: middle;
                  border: 1px solid #aaa; margin-right: 4px; }
section.view { display: none; }
section.view.active { display: block; }
.panel-grid { display: grid; grid-template-columns: repeat(auto-fit,
              minmax(560px, 1fr)); gap: 22px; margin: 14px 0 0 0; }
.panel { background: #fff; border: 1px solid #ddd; border-radius: 4px;
         padding: 10px 14px; min-width: 0; overflow-x: auto; }
.panel h3 { margin: 0 0 2px 0; font-size: 1.05em; color: #222; }
.panel .panel-meta { color: #666; font-size: 0.85em; margin-bottom: 8px; }
.panel .panel-meta code { background: #f0f0f0; padding: 0 4px;
                           border-radius: 3px; }
.view-toggle { display: flex; gap: 8px; margin: 1.2em 0 0.4em 0; }
.view-toggle button { font: inherit; background: white; border: 1px solid #aaa;
                      padding: 6px 14px; border-radius: 14px; cursor: pointer;
                      color: #225; }
.view-toggle button:hover { background: #eef; border-color: #225; }
.view-toggle button.active { background: #225; color: white; border-color: #225; }
"""

JS = """
function showView(n, btn) {
  for (const s of document.querySelectorAll('section.view')) {
    s.classList.toggle('active', s.dataset.view === String(n));
  }
  for (const b of document.querySelectorAll('.view-toggle button')) {
    b.classList.toggle('active', b.dataset.view === String(n));
  }
}
"""


def compute_panel(path: Path, axis: str, include_unanimous: bool) -> dict:
    """Load + filter one CSV and compute its agreement & Spearman matrices.
    Returns a dict ready to feed into build_html as one panel."""
    scenarios, rates, variant_keys = load_yes_rate_matrix(
        path, axis, require_complete=False)
    n_total = len(scenarios)
    unanimous_count = 0
    if not include_unanimous:
        unanimous_set = {sid for sid in scenarios
                         if is_unanimous(rates, sid, variant_keys)}
        unanimous_count = len(unanimous_set)
        scenarios = [sid for sid in scenarios if sid not in unanimous_set]
        rates = {v: {sid: rates[v][sid] for sid in scenarios if sid in rates[v]}
                 for v in variant_keys}
    if len(scenarios) < 5:
        raise SystemExit(
            f"error: only {len(scenarios)} scenarios remain after filtering "
            f"in {path} — need at least 5. Try --include-unanimous.")
    with path.open(newline="") as f:
        model = next(
            (r.get("model", "") for r in csv.DictReader(f) if r.get("model")), "")
    return {
        "path": path, "model": model,
        "n_total": n_total, "n_used": len(scenarios),
        "unanimous_count": unanimous_count,
        "agree": agreement_matrix(scenarios, rates, variant_keys),
        "corr": spearman_matrix(scenarios, rates, variant_keys),
    }


def _panel_meta(panel: dict) -> str:
    note = (f"{panel['unanimous_count']} unanimous-consensus dropped"
            if panel['unanimous_count'] else
            "unanimous-consensus scenarios included")
    return (
        f'<div class="panel-meta">Source: <code>{esc(panel["path"])}</code><br>'
        f'{panel["n_used"]} of {panel["n_total"]} scenarios used '
        f'({esc(note)}; partial-coverage scenarios included; '
        f'pairs use pairwise complete observations)</div>'
    )


def _render_panel_grid(panels: list[dict], matrix_key: str,
                       display_labels: list[str], color_fn) -> str:
    """Build a horizontal grid of one matrix per model panel."""
    tiles = []
    for panel in panels:
        tiles.append(
            '<div class="panel">'
            f'<h3>{esc(panel["model"] or "(unknown model)")}</h3>'
            f'{_panel_meta(panel)}'
            f'{render_matrix(panel[matrix_key], display_labels, color_fn)}'
            '</div>'
        )
    return '<div class="panel-grid">' + "".join(tiles) + '</div>'


def build_html(panels: list[dict], axis: str,
               display_labels: list[str]) -> str:
    n = len(display_labels)
    axis_title = {"both": "15 race × income variants",
                  "race": "5 race categories (pooled across income)",
                  "income": "3 income categories (pooled across race)"}[axis]
    parts: list[str] = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Variant trends — {esc(axis_title)}</title>
<style>{CSS}</style>
<script>{JS}</script></head><body>""")
    parts.append(f"<h1>Variant trend analysis &mdash; {esc(axis_title)}</h1>")
    if len(panels) == 1:
        models_line = f'Model: <code>{esc(panels[0]["model"])}</code>'
    else:
        models_line = ('Models (side-by-side): '
                       + ' &middot; '.join(
                           f'<code>{esc(p["model"])}</code>' for p in panels))
    parts.append(
        f'<div class="meta">{models_line} &middot; '
        f'Axis: <code>{esc(axis)}</code></div>'
    )

    parts.append(
        '<div class="view-toggle">'
        '<button type="button" class="active" data-view="1" '
        'onclick="showView(1, this)">1. Pairwise agreement</button>'
        '<button type="button" data-view="2" '
        'onclick="showView(2, this)">2. Spearman correlation</button>'
        '</div>'
    )

    # Method 1
    parts.append('<section class="view active" data-view="1">')
    parts.append("<h2>1. Pairwise agreement matrix</h2>")
    parts.append(
        '<div class="note">'
        "<p><strong>What this shows:</strong> For every pair of demographic "
        "variants, the fraction of scenarios on which both gave the same "
        "Yes/No verdict.</p>"
        "<p><strong>How it&rsquo;s computed:</strong></p>"
        "<ul>"
        "<li>For each variant <em>v</em> on each scenario <em>s</em>, "
        "yes-rate(<em>v</em>, <em>s</em>) = #Yes / (#Yes + #No) across the "
        "n = 5 replicates.</li>"
        "<li>majority(<em>v</em>, <em>s</em>) = Yes if yes-rate &gt; 0.5, "
        "No if &lt; 0.5, undefined if exactly 0.5 "
        "(exactly-tied variants are excluded from that pair&rsquo;s count).</li>"
        "<li>Cell (<em>v<sub>i</sub></em>, <em>v<sub>j</sub></em>) = "
        "#scenarios where both majorities are defined and equal &divide; "
        "#scenarios where both are defined.</li>"
        "<li>Diagonal is trivially 1.00 &mdash; a variant always agrees with "
        "itself.</li>"
        "</ul>"
        f"<p><strong>Row mean (excl. self):</strong> Average of the {n - 1} "
        "off-diagonal cells in each row. Quantifies how aligned this variant "
        "is with the others on average. A <em>low</em> row mean flags an "
        "outlier variant whose majorities frequently disagree with the rest; "
        "a <em>high</em> row mean indicates a variant that sits in the "
        "consensus.</p>"
        "<p><strong>How to read it:</strong> Hot cells = answer together, "
        "cold cells = answer opposite. The row-mean column condenses each "
        "variant into a single &ldquo;how central is this variant&rdquo; "
        "score on the same color scale as the matrix.</p>"
        "</div>"
    )
    parts.append(
        '<div class="legend">'
        '<span><span class="swatch" style="background:hsl(220,60%,100%)"></span>0.0 (always disagree)</span>'
        '<span><span class="swatch" style="background:hsl(220,60%,72%)"></span>0.5</span>'
        '<span><span class="swatch" style="background:hsl(220,60%,45%)"></span>1.0 (always agree)</span>'
        '</div>'
    )
    parts.append(_render_panel_grid(panels, "agree", display_labels,
                                    hsl_for_agreement))
    parts.append("</section>")

    # Method 2
    parts.append('<section class="view" data-view="2">')
    parts.append("<h2>2. Spearman rank correlation</h2>")
    parts.append(
        '<div class="note">'
        "<p><strong>What this shows:</strong> For every pair of variants, "
        "the rank correlation between their per-scenario yes-rates. Captures "
        "<em>partial</em> agreement that method 1&rsquo;s majority-only view "
        "discards.</p>"
        "<p><strong>How it&rsquo;s computed:</strong></p>"
        "<ul>"
        "<li>Treat each variant as a length-<em>N</em> vector of yes-rates "
        "(one value per scenario).</li>"
        "<li>Replace each value with its rank (1 = lowest yes-rate, "
        "<em>N</em> = highest). Ties receive mid-ranks &mdash; e.g. three "
        "tied values share rank 5 each, not 4/5/6.</li>"
        "<li>Cell (<em>v<sub>i</sub></em>, <em>v<sub>j</sub></em>) = Pearson "
        "correlation between <em>v<sub>i</sub></em>&rsquo;s rank vector and "
        "<em>v<sub>j</sub></em>&rsquo;s rank vector.</li>"
        "<li>Range: &minus;1 (perfectly opposite movement) through 0 "
        "(no monotonic relationship) to +1 (perfectly aligned movement). "
        "Diagonal is 1.00.</li>"
        "</ul>"
        "<p><strong>Why both this <em>and</em> method 1:</strong> Variant A "
        "with yes-rates (0.8, 0.6, 0.2) and variant B with (0.7, 0.5, 0.1) "
        "have identical Yes/Yes/No majorities, so method 1 scores them "
        "100%. Spearman additionally detects that their yes-rates move in "
        "lockstep across scenarios &mdash; a stronger statement than "
        "&ldquo;they happened to land on the same side of 0.5 three "
        "times.&rdquo;</p>"
        f"<p><strong>Row mean (excl. self):</strong> Average of the {n - 1} "
        "off-diagonal Spearman values in each row. Quantifies how strongly "
        "this variant&rsquo;s yes-rate co-varies with the others on average. "
        "A low row mean flags a variant that marches to its own beat; a "
        "high one indicates a variant whose answer pattern broadly tracks "
        "the rest.</p>"
        "<p><strong>How to read it:</strong> Blue cells = answer together "
        "(positive &rho;); red cells = answer opposite (negative &rho;); "
        "pale cells = independent.</p>"
        "</div>"
    )
    parts.append(
        '<div class="legend">'
        '<span><span class="swatch" style="background:hsl(0,60%,45%)"></span>&minus;1</span>'
        '<span><span class="swatch" style="background:hsl(0,60%,100%)"></span>0</span>'
        '<span><span class="swatch" style="background:hsl(220,60%,45%)"></span>+1</span>'
        '</div>'
    )
    parts.append(_render_panel_grid(panels, "corr", display_labels,
                                    hsl_for_corr))
    parts.append("</section>")

    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("results_csv", help="Path to results CSV.")
    p.add_argument("--compare", default=None, metavar="CSV",
                   help="Optional second results CSV. When provided, both "
                        "models' matrices are rendered side-by-side in each "
                        "plot section.")
    p.add_argument("--output", default=None,
                   help="Output HTML path. Default: "
                        "reports/variant_trends_<model>[_vs_<model2>]"
                        "[_axis-<race|income>].html.")
    p.add_argument("--include-unanimous", action="store_true",
                   help="Keep scenarios where every variant gives the same "
                        "majority answer (dropped by default — no demographic "
                        "signal but they pull every pairwise correlation toward "
                        "each other).")
    p.add_argument("--axis", choices=("both", "race", "income"), default="both",
                   help="Variant axis. both = 15 (race, income) variants "
                        "(default). race = 5 race categories, pooling across "
                        "income. income = 3 income categories, pooling across "
                        "race.")
    args = p.parse_args()

    paths = [Path(args.results_csv)]
    if args.compare:
        paths.append(Path(args.compare))
    for p_ in paths:
        if not p_.exists():
            print(f"error: results CSV not found: {p_}", file=sys.stderr)
            sys.exit(2)

    _, display_labels = axis_categories(args.axis)
    panels = [compute_panel(p_, args.axis, args.include_unanimous)
              for p_ in paths]
    for panel in panels:
        if panel["unanimous_count"]:
            print(
                f"{panel['path'].name}: dropped {panel['unanimous_count']} "
                f"unanimous-consensus scenarios "
                f"({panel['unanimous_count'] / panel['n_total']:.0%} of "
                f"{panel['n_total']}); {panel['n_used']} remain.",
                file=sys.stderr,
            )

    if args.output:
        requested = Path(args.output)
    elif len(panels) == 1:
        requested = default_output_path(panels[0]["model"], args.axis)
    else:
        a = _sanitize_for_filename(panels[0]["model"] or "a")
        b = _sanitize_for_filename(panels[1]["model"] or "b")
        suffix = f"_axis-{args.axis}" if args.axis != "both" else ""
        requested = (Path(__file__).resolve().parent / "reports"
                     / f"variant_trends_{a}_vs_{b}{suffix}.html")
    output_path = resolve_non_clobbering(requested)
    if output_path != requested:
        print(f"warning: {requested} exists; writing to {output_path} instead",
              file=sys.stderr)

    html = build_html(panels, args.axis, display_labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"wrote trend report ({len(panels)} model"
          f"{'s' if len(panels) != 1 else ''}, axis={args.axis}) "
          f"to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
