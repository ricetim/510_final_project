#!/usr/bin/env python3
"""Gallery of contested scenarios as per-scenario 5x3 yes-rate heatmaps.

Filters the input CSV to contested scenarios (where the 15 demographic
variants don't unanimously agree on Yes vs No), then renders each one as
a tile: the full original scenario text, a small race x income heatmap
showing the yes-rate for every (race, income) cell, and a header line
with the scenario id + per-scenario spread + the Yes/No vote count.

A high-contrast color key at the top of the report explains what each
color means.

Usage:
    python contested_gallery_report.py <results_csv>
        [--output report.html]
        [--top-n N]               # default: all contested
        [--sort spread|id]        # default: spread (desc)
        [--include-unanimous]     # default: filter out

Default output: reports/contested_gallery_<model>.html, anchored to the
script's parent dir (not CWD), no-clobber.
"""

import argparse
import csv
import html as _html
import sys
from pathlib import Path

from collections import defaultdict

from variant_trends_report import (
    RACES, INCOMES, VARIANTS, majority,
    _sanitize_for_filename, resolve_non_clobbering,
)


def esc(v) -> str:
    return _html.escape(str(v) if v is not None else "")


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

def yr_color(yr: float) -> str:
    """Diverging: red at 0 (No), white at 0.5, blue at 1 (Yes)."""
    if yr is None or yr != yr:
        return "#eee"
    yr = max(0.0, min(1.0, yr))
    if yr >= 0.5:
        s = (yr - 0.5) * 2
        return f"hsl(220, 60%, {100 - 55 * s:.0f}%)"
    s = (0.5 - yr) * 2
    return f"hsl(0, 60%, {100 - 55 * s:.0f}%)"


def text_color(yr: float) -> str:
    if yr is None or yr != yr:
        return "#888"
    return "#fff" if abs(yr - 0.5) > 0.4 else "#222"


# ---------------------------------------------------------------------------
# Partial-coverage loader (local — variant_trends_report's loader drops any
# scenario where one of the 15 cells has no Yes/No, but for the gallery we
# WANT to show those so the user can see WHICH cells are missing and why).
# ---------------------------------------------------------------------------

def load_partial_yes_rates(path) -> tuple[list[str], dict]:
    """Return (scenarios_sorted, rates) without dropping incomplete scenarios.

    A scenario is included if at least ONE (race, income) cell has at least
    one Yes/No reply. Cells without any Yes/No simply don't appear in the
    inner dict — callers should test `sid in rates[v]` before reading.
    """
    raw: dict = defaultdict(lambda: defaultdict(list))
    all_sids: set[str] = set()
    import csv as _csv  # local alias to avoid shadowing
    with open(path, newline="") as f:
        for r in _csv.DictReader(f):
            if r["answer"] not in ("Yes", "No"):
                continue
            if r["race_variant"] not in RACES or r["income_variant"] not in INCOMES:
                continue
            v = (r["race_variant"], r["income_variant"])
            raw[v][r["scenario_id"]].append(r["answer"])
            all_sids.add(r["scenario_id"])

    rates: dict = {v: {} for v in VARIANTS}
    for v in VARIANTS:
        for sid, answers in raw[v].items():
            rates[v][sid] = sum(1 for a in answers if a == "Yes") / len(answers)
    return sorted(all_sids), rates


def coverage_of(rates: dict, sid: str) -> int:
    """How many of the 15 cells have any Yes/No data for this scenario."""
    return sum(1 for v in VARIANTS if sid in rates[v])


def is_unanimous_partial(rates: dict, sid: str) -> bool:
    """All covered cells give the same Yes/No majority on this scenario.

    Ambiguous cells (yes-rate exactly 0.5) prevent unanimity. If a scenario
    has zero covered cells, returns False (we want to surface it as a data
    issue, not silently drop it).
    """
    seen: set[str] = set()
    for v in VARIANTS:
        if sid not in rates[v]:
            continue
        m = majority(rates[v][sid])
        if m is None:
            return False
        seen.add(m)
    if not seen:
        return False
    return len(seen) == 1


def spread_of(rates: dict, sid: str) -> float:
    """Max-minus-min yes-rate across the cells with data (0.0 if 0 or 1 cell)."""
    ys = [rates[v][sid] for v in VARIANTS if sid in rates[v]]
    if len(ys) < 2:
        return 0.0
    return max(ys) - min(ys)


def render_coverage_badge(coverage: int) -> str:
    if coverage == 15:
        return ('<span class="cov-badge cov-full">'
                '15 / 15 covered</span>')
    if coverage >= 12:
        return (f'<span class="cov-badge cov-partial">'
                f'{coverage} / 15 covered &mdash; partial</span>')
    return (f'<span class="cov-badge cov-low">'
            f'{coverage} / 15 covered &mdash; low coverage</span>')


# ---------------------------------------------------------------------------
# SVG mini-heatmap (race columns x income rows, with axis labels)
# ---------------------------------------------------------------------------

def mini_heatmap_svg(rates, sid, w=300, h=140) -> str:
    cw = w / 5
    ch = h / 3
    # Reserve 16px above (race labels) and 60px on the right (income labels)
    parts = [f'<svg viewBox="-2 -18 {w + 60} {h + 24}" '
             f'xmlns="http://www.w3.org/2000/svg" '
             f'font-family="sans-serif" font-size="10">']
    # Column headers (races)
    for ri, race in enumerate(RACES):
        cx = ri * cw + cw / 2
        parts.append(
            f'<text x="{cx:.1f}" y="-4" text-anchor="middle" '
            f'font-size="10" fill="#444">{esc(race)}</text>'
        )
    # Cells — each one is a clickable <g> with data attrs that the JS filter reads.
    # Cells without Yes/No data render as a gray dashed square ("—") so the
    # missing-data structure is visible at a glance.
    for ii, inc in enumerate(INCOMES):
        for ri, race in enumerate(RACES):
            x, y = ri * cw, ii * ch
            v = (race, inc)
            has_data = sid in rates[v]
            if has_data:
                yr = rates[v][sid]
                parts.append(
                    f'<g class="cell-click" data-race="{esc(race)}" '
                    f'data-income="{esc(inc)}" onclick="filterTile(this)">'
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw:.1f}" height="{ch:.1f}" '
                    f'fill="{yr_color(yr)}" stroke="#fff" stroke-width="1"/>'
                    f'<text x="{x + cw / 2:.1f}" y="{y + ch / 2 + 4:.1f}" '
                    f'text-anchor="middle" fill="{text_color(yr)}" font-size="11" '
                    f'font-variant-numeric="tabular-nums" pointer-events="none">{yr:.2f}</text>'
                    f'</g>'
                )
            else:
                # Missing cell: gray fill + a diagonal line + em-dash.
                # Still clickable so the user can filter to the (race, income)
                # group and see the error/refusal explanations underneath.
                parts.append(
                    f'<g class="cell-click cell-missing" data-race="{esc(race)}" '
                    f'data-income="{esc(inc)}" onclick="filterTile(this)">'
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw:.1f}" height="{ch:.1f}" '
                    f'fill="#ececec" stroke="#fff" stroke-width="1"/>'
                    f'<line x1="{x + 2:.1f}" y1="{y + 2:.1f}" '
                    f'x2="{x + cw - 2:.1f}" y2="{y + ch - 2:.1f}" '
                    f'stroke="#cfcfcf" stroke-width="1" pointer-events="none"/>'
                    f'<text x="{x + cw / 2:.1f}" y="{y + ch / 2 + 4:.1f}" '
                    f'text-anchor="middle" fill="#888" font-size="11" '
                    f'pointer-events="none">&#8212;</text>'
                    f'</g>'
                )
        short = inc.replace(" income", "")
        parts.append(
            f'<text x="{w + 6:.1f}" y="{ii * ch + ch / 2 + 4:.1f}" '
            f'text-anchor="start" font-size="11" fill="#444">{esc(short)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Per-scenario tile + color-key legend
# ---------------------------------------------------------------------------

def vote_counts(rates, sid) -> tuple[int, int, int, int]:
    """Return (n_yes, n_no, n_ambig, n_missing) across the 15 variants."""
    n_yes = n_no = n_ambig = n_missing = 0
    for v in VARIANTS:
        if sid not in rates[v]:
            n_missing += 1
            continue
        m = majority(rates[v][sid])
        if m == "Yes":
            n_yes += 1
        elif m == "No":
            n_no += 1
        else:
            n_ambig += 1
    return n_yes, n_no, n_ambig, n_missing


def render_color_key(steps: int = 11, swatch_w: int = 36, swatch_h: int = 26) -> str:
    """Horizontal swatch strip 0.0 -> 1.0 with labeled ticks and a legend."""
    parts = ['<div class="key">']
    parts.append('<div class="key-title">Color key &mdash; cell yes-rate '
                 '(<em>fraction of replicates that answered Yes</em>)</div>')
    parts.append('<div class="swatch-row">')
    for i in range(steps):
        yr = i / (steps - 1)
        bg = yr_color(yr)
        fg = text_color(yr)
        parts.append(
            f'<div class="swatch" style="background:{bg}; color:{fg}; '
            f'width:{swatch_w}px; height:{swatch_h}px;">{yr:.1f}</div>'
        )
    parts.append('</div>')
    parts.append(
        '<div class="key-legend">'
        '<span><strong>0.00</strong> &mdash; every replicate answered '
        '<strong style="color:#b13030">No</strong> '
        '(unacceptable behavior)</span>'
        '<span><strong>0.50</strong> &mdash; split / ambiguous '
        '(group has no clear majority)</span>'
        '<span><strong>1.00</strong> &mdash; every replicate answered '
        '<strong style="color:#1a4a8a">Yes</strong> '
        '(acceptable behavior)</span>'
        '</div>'
    )
    parts.append(
        '<div class="key-legend">'
        '<span><span class="swatch" style="background:#ececec; color:#888;'
        ' width:36px; height:18px; display:inline-flex; align-items:center;'
        ' justify-content:center;">&mdash;</span> '
        '<strong>missing cell</strong> &mdash; no Yes/No replicate (cell was '
        'all errors, blanks, or refusals; or the variant was queued but never '
        'completed)</span>'
        '<span><span class="cov-badge cov-full">15 / 15</span> '
        '<span class="cov-badge cov-partial">13 / 15 partial</span> '
        '<span class="cov-badge cov-low">8 / 15 low</span> '
        '&mdash; coverage badges shown next to each scenario&rsquo;s spread.</span>'
        '</div>'
    )
    parts.append(
        '<div class="click-hint">'
        '<strong>Tip:</strong> click any cell in a tile (including missing ones) '
        'to filter that scenario&rsquo;s explanations to just that '
        '<code>(race / income)</code> group. Click the same cell again, or '
        'the <code>clear filter</code> button, to show all rows again.'
        '</div>'
    )
    parts.append('</div>')
    return "".join(parts)


def render_explanations(explanations: list[tuple]) -> str:
    """Render a <details> block with one row per explanation-bearing replicate.

    explanations: list of (race, income, replicate_idx, answer, explanation).
    Sorted in canonical order (race, income, replicate) before rendering.
    Returns "" if the list is empty.
    """
    if not explanations:
        return ""
    # Stable canonical order: by RACES index, then INCOMES index, then replicate.
    race_idx = {r: i for i, r in enumerate(RACES)}
    inc_idx = {i: k for k, i in enumerate(INCOMES)}

    def sort_key(t):
        race, inc, rep, _, _ = t
        try:
            rep_n = int(rep)
        except (TypeError, ValueError):
            rep_n = -1
        return (race_idx.get(race, 99), inc_idx.get(inc, 99), rep_n)

    rows_html = []
    for race, inc, rep, answer, exp in sorted(explanations, key=sort_key):
        if answer == "Yes":
            ans_cls = "ans-yes"
        elif answer == "No":
            ans_cls = "ans-no"
        else:
            ans_cls = "ans-other"
        # Each replicate is rendered as two TRs sharing data-race/data-income:
        # a 4-cell metadata row, then a single-cell exp-row that colspans the
        # whole table width so the prose isn't crammed into a narrow column.
        rows_html.append(
            f'<tr class="meta-row" data-race="{esc(race)}" data-income="{esc(inc)}">'
            f'<td class="r-race">{esc(race)}</td>'
            f'<td class="r-inc">{esc(inc.replace(" income", ""))}</td>'
            f'<td class="r-rep">{esc(rep)}</td>'
            f'<td class="r-ans {ans_cls}">{esc(answer or "—")}</td>'
            f'</tr>'
            f'<tr class="exp-row" data-race="{esc(race)}" data-income="{esc(inc)}">'
            f'<td colspan="4" class="r-exp">{esc(exp)}</td>'
            f'</tr>'
        )
    return (
        '<details class="expl">'
        f'<summary>Show {len(explanations)} explanation'
        f'{"s" if len(explanations) != 1 else ""}</summary>'
        '<table class="expl-tbl"><thead><tr>'
        '<th>race</th><th>income</th><th>#</th><th>answer</th>'
        '</tr></thead><tbody>'
        + "".join(rows_html) +
        '</tbody></table></details>'
    )


def render_tile(sid, rates, orig, spreads,
                explanations: list[tuple]) -> str:
    n_yes, n_no, n_ambig, n_missing = vote_counts(rates, sid)
    coverage = 15 - n_missing
    text = orig.get(sid, "") or "(no original scenario text available)"
    ambig_part = f" &middot; {n_ambig} split" if n_ambig else ""
    return (
        f'<div class="cell">'
        f'<div class="cap">'
        f'<strong class="sid">{esc(sid)}</strong>'
        f'<span class="sep">&middot;</span>'
        f'spread <strong>{spreads[sid]:.2f}</strong>'
        f'<span class="sep">&middot;</span>'
        f'<span class="vote">'
        f'<span class="yes-pill">{n_yes} Yes</span>'
        f'<span class="no-pill">{n_no} No</span>'
        f'{ambig_part}'
        f'</span>'
        f'{render_coverage_badge(coverage)}'
        f'</div>'
        f'<div class="orig">{esc(text)}</div>'
        f'{mini_heatmap_svg(rates, sid)}'
        f'{render_explanations(explanations)}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1400px; margin: 1.5em auto; padding: 0 1em; color: #222; }
h1 { font-size: 1.55em; margin-bottom: 0.2em; }
.meta { color: #666; font-size: 0.9em; margin-bottom: 1em; }

.key { background: #f7f7f7; border: 1px solid #ddd; border-radius: 4px;
       padding: 12px 16px; margin: 0.5em 0 1.5em 0; }
.key-title { font-size: 0.95em; margin-bottom: 8px; color: #333; }
.key-title em { color: #555; font-style: normal; font-weight: 500; }
.swatch-row { display: flex; gap: 0; }
.swatch { display: flex; align-items: center; justify-content: center;
          font-size: 0.78em; font-variant-numeric: tabular-nums;
          border-right: 1px solid white; }
.swatch:first-child { border-top-left-radius: 3px; border-bottom-left-radius: 3px; }
.swatch:last-child { border-right: none;
                     border-top-right-radius: 3px; border-bottom-right-radius: 3px; }
.key-legend { display: flex; justify-content: space-between; margin-top: 8px;
              font-size: 0.85em; color: #555; gap: 1.5em; flex-wrap: wrap; }
.key-legend strong { color: #222; }

.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(440px, 1fr));
           gap: 16px; align-items: start; }
.gallery .cell { border: 1px solid #ddd; padding: 10px 12px 8px 12px;
                 background: #fcfcfc; border-radius: 4px; }
.gallery .cap { font-size: 0.85em; color: #333; margin-bottom: 8px;
                line-height: 1.6em; }
.gallery .cap .sid { font-size: 1em; color: #111; }
.gallery .cap .sep { color: #bbb; margin: 0 6px; }
.gallery .cap .vote { display: inline-flex; gap: 6px; align-items: center;
                      flex-wrap: wrap; }
.gallery .cap .yes-pill { background: hsl(220, 60%, 88%); color: #1a4a8a;
                          padding: 1px 8px; border-radius: 9px; font-size: 0.9em; }
.gallery .cap .no-pill { background: hsl(0, 60%, 92%); color: #b13030;
                         padding: 1px 8px; border-radius: 9px; font-size: 0.9em; }
.gallery .orig { font-size: 0.92em; color: #222; line-height: 1.5;
                 margin: 0 0 10px 0; white-space: pre-wrap; }
.gallery svg { width: 100%; height: auto; }

details.expl { margin-top: 10px; border-top: 1px solid #eee; padding-top: 8px; }
details.expl > summary { cursor: pointer; font-size: 0.85em; color: #225;
                         padding: 4px 0; user-select: none; }
details.expl > summary:hover { color: #003; }
details.expl[open] > summary { font-weight: 600; }
table.expl-tbl { width: 100%; border-collapse: collapse; margin-top: 6px;
                 font-size: 0.85em; }
table.expl-tbl th { background: #f0f0f0; text-align: left; font-weight: 600;
                    color: #333; padding: 4px 8px; border: 1px solid #e3e3e3; }
table.expl-tbl tr.meta-row > td { padding: 4px 8px; vertical-align: middle;
                                  border-left: 1px solid #e3e3e3;
                                  border-right: 1px solid #e3e3e3;
                                  border-top: 1px solid #e3e3e3; }
table.expl-tbl td.r-race { width: 30%; color: #444; }
table.expl-tbl td.r-inc  { width: 22%; color: #444; }
table.expl-tbl td.r-rep  { width: 10%; text-align: right;
                            font-variant-numeric: tabular-nums; color: #888; }
table.expl-tbl td.r-ans  { width: 22%; font-weight: 600; text-align: center; }
table.expl-tbl td.ans-yes { color: #1a8830; background: hsl(220, 60%, 96%); }
table.expl-tbl td.ans-no  { color: #b13030; background: hsl(0, 60%, 96%); }
table.expl-tbl td.ans-other { color: #888; font-style: italic; }
table.expl-tbl tr.exp-row > td.r-exp {
    background: #fdfdfd; border: 1px solid #e3e3e3; border-top: none;
    padding: 6px 10px 10px 10px; line-height: 1.5; color: #333;
    white-space: pre-wrap; word-wrap: break-word;
    border-bottom: 2px solid #e0e0e0; }

g.cell-click { cursor: pointer; }
g.cell-click:hover rect { stroke: #444; stroke-width: 2; }
g.cell-click.active rect { stroke: #111; stroke-width: 3; }
g.cell-missing:hover rect { stroke: #888; }
g.cell-missing.active rect { stroke: #444; }

.cov-badge { display: inline-block; padding: 1px 8px; border-radius: 9px;
             font-size: 0.85em; margin-left: 6px; vertical-align: middle;
             white-space: nowrap; }
.cov-badge.cov-full { background: hsl(120, 50%, 92%); color: #2a6b2a; }
.cov-badge.cov-partial { background: hsl(45, 75%, 88%); color: #8a6a00;
                         font-weight: 500; }
.cov-badge.cov-low { background: hsl(0, 70%, 92%); color: #b13030;
                     font-weight: 600; }
button.clear-link { background: white; border: 1px solid #aaa;
                    padding: 1px 8px; border-radius: 9px; font-size: 0.88em;
                    color: #225; cursor: pointer; font-family: inherit;
                    margin-left: 4px; }
button.clear-link:hover { background: #eef; border-color: #225; }
.click-hint { color: #555; font-size: 0.88em; margin-top: 6px; }
.click-hint code { background: #eef; padding: 0 4px; border-radius: 3px;
                   font-size: 0.95em; }
"""

JS = """
function filterTile(g) {
  const race = g.dataset.race;
  const income = g.dataset.income;
  const tile = g.closest('.cell');
  const det = tile.querySelector('details.expl');
  if (!det) return;
  const tbody = det.querySelector('table.expl-tbl tbody');
  if (!tbody) { det.open = true; return; }
  const summary = det.querySelector('summary');
  if (!summary.dataset.allLabel) summary.dataset.allLabel = summary.innerHTML;

  const filterKey = race + '|' + income;
  if (det.dataset.filter === filterKey) {
    // Same cell clicked twice — collapse the details AND clear the filter,
    // so a second click is the natural "I'm done with this group" gesture.
    // (The 'clear filter' button beside the summary still just clears
    // without collapsing — leaves the table visible for further exploration.)
    clearTileFilter(tile);
    det.open = false;
    return;
  }

  det.open = true;
  det.dataset.filter = filterKey;
  let visible = 0;
  let total = 0;
  for (const tr of tbody.rows) {
    const isMeta = tr.classList.contains('meta-row');
    if (isMeta) total++;
    if (tr.dataset.race === race && tr.dataset.income === income) {
      tr.style.display = '';
      if (isMeta) visible++;
    } else {
      tr.style.display = 'none';
    }
  }

  for (const el of tile.querySelectorAll('.cell-click.active')) {
    el.classList.remove('active');
  }
  g.classList.add('active');

  const incShort = income.replace(' income', '');
  const noun = (total === 1) ? 'explanation' : 'explanations';
  summary.innerHTML = 'Showing <strong>' + visible + '</strong> of '
    + total + ' ' + noun
    + ' \\u2014 filter: <strong>' + race + ' / ' + incShort + '</strong> '
    + '<button type="button" class="clear-link" '
    + 'onclick="event.preventDefault(); event.stopPropagation(); '
    + 'clearTileFilter(this.closest(&quot;.cell&quot;));">clear filter</button>';
}

function clearTileFilter(tile) {
  const det = tile.querySelector('details.expl');
  if (!det) return;
  det.dataset.filter = '';
  const tbody = det.querySelector('table.expl-tbl tbody');
  if (tbody) {
    for (const tr of tbody.rows) tr.style.display = '';
  }
  const summary = det.querySelector('summary');
  if (summary.dataset.allLabel) summary.innerHTML = summary.dataset.allLabel;
  for (const el of tile.querySelectorAll('.cell-click.active')) {
    el.classList.remove('active');
  }
}
"""


def build_html(results_path, model,
               n_in_csv, n_full, n_partial, n_candidates,
               shown_sids, rates, orig, spreads,
               explanations_by_sid: dict,
               sort_mode, include_unanimous) -> str:
    tiles = "".join(
        render_tile(sid, rates, orig, spreads,
                    explanations_by_sid.get(sid, []))
        for sid in shown_sids
    )
    n_with_expl = sum(1 for sid in shown_sids if explanations_by_sid.get(sid))
    expl_note = (
        f" &middot; {n_with_expl} have expandable explanations"
        if n_with_expl else
        " &middot; no explanations in this run"
    )
    filter_note = (" &middot; <code>--include-unanimous</code> active "
                   "(fully-covered unanimous scenarios kept)"
                   if include_unanimous else
                   " &middot; fully-covered unanimous scenarios filtered out")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Contested-scenario gallery</title>
<style>{CSS}</style>
<script>{JS}</script>
</head><body>
<h1>Contested-scenario gallery</h1>
<div class="meta">Source: <code>{esc(results_path)}</code> &middot;
Model: <code>{esc(model)}</code><br>
<strong>{n_in_csv}</strong> scenarios in CSV
&mdash; {n_full} full-coverage, {n_partial} partial
&middot; showing <strong>{len(shown_sids)}</strong>
of {n_candidates} candidates after filtering
&middot; sort: <code>{esc(sort_mode)}</code>{filter_note}{expl_note}</div>

{render_color_key()}

<div class="gallery">{tiles}</div>
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
    p.add_argument("results_csv")
    p.add_argument("--output", default=None,
                   help="Output HTML path. Default: "
                        "reports/contested_gallery_<model>.html.")
    p.add_argument("--top-n", type=int, default=None,
                   help="Show only the top-N most contested scenarios. "
                        "Default: show all contested.")
    p.add_argument("--sort", choices=("spread", "id"), default="spread",
                   help="spread (default) = max-min yes-rate desc; "
                        "id = scenario_id ascending.")
    p.add_argument("--include-unanimous", action="store_true",
                   help="Keep unanimous scenarios (every cell agrees, "
                        "tiles will be all one color; off by default).")
    args = p.parse_args()

    results_path = Path(args.results_csv)
    if not results_path.exists():
        print(f"error: results CSV not found: {results_path}", file=sys.stderr)
        sys.exit(2)

    all_sids, rates = load_partial_yes_rates(results_path)
    n_in_csv = len(all_sids)

    coverage_by_sid = {s: coverage_of(rates, s) for s in all_sids}
    n_full = sum(1 for c in coverage_by_sid.values() if c == 15)
    n_partial = n_in_csv - n_full

    # Default filter: drop fully-covered unanimous scenarios — they carry no
    # signal. Partially-covered scenarios are ALWAYS kept (the data-quality
    # issue itself is the signal we want surfaced).
    if args.include_unanimous:
        candidates = list(all_sids)
    else:
        candidates = [
            s for s in all_sids
            if not (coverage_by_sid[s] == 15 and is_unanimous_partial(rates, s))
        ]
    n_candidates = len(candidates)

    if not candidates:
        print("error: no scenarios to show after filtering.", file=sys.stderr)
        sys.exit(2)

    spreads = {s: spread_of(rates, s) for s in candidates}

    if args.sort == "spread":
        # Primary: spread desc.
        # Secondary: more-damaged (more missing) first within tied spread —
        # surfaces partial-coverage scenarios near the top of their band.
        # Tertiary: scenario_id asc, for stable ordering.
        candidates.sort(key=lambda s: (-spreads[s],
                                       -(15 - coverage_by_sid[s]),
                                       s))
    else:
        candidates.sort()

    shown = candidates if args.top_n is None else candidates[: args.top_n]

    with results_path.open(newline="") as f:
        raw_rows = list(csv.DictReader(f))
    shown_set = set(shown)
    orig = {r["scenario_id"]: r.get("original_scenario", "")
            for r in raw_rows if r["scenario_id"] in shown_set}
    model = next((r.get("model", "") for r in raw_rows if r.get("model")), "")

    # Collect explanation-bearing rows per scenario.
    explanations_by_sid: dict[str, list[tuple]] = {}
    for r in raw_rows:
        sid = r.get("scenario_id", "")
        if sid not in shown_set:
            continue
        exp = (r.get("explanation") or "").strip()
        if not exp:
            continue
        explanations_by_sid.setdefault(sid, []).append((
            r.get("race_variant", ""),
            r.get("income_variant", ""),
            r.get("replicate_idx", ""),
            r.get("answer", ""),
            exp,
        ))

    requested = Path(args.output) if args.output else (
        Path(__file__).resolve().parent / "reports"
        / f"contested_gallery_{_sanitize_for_filename(model)}.html"
    )
    output_path = resolve_non_clobbering(requested)
    if output_path != requested:
        print(f"warning: {requested} exists; writing to {output_path} instead",
              file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_html(results_path, model,
                   n_in_csv, n_full, n_partial, n_candidates,
                   shown, rates, orig, spreads,
                   explanations_by_sid,
                   args.sort, args.include_unanimous),
        encoding="utf-8",
    )
    print(f"wrote gallery of {len(shown)} scenarios to {output_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
