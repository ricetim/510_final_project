#!/usr/bin/env python3
"""Cross-model contested-scenario gallery.

Takes one or more Claude CSVs and one or more OpenAI CSVs. For each
scenario present in BOTH sides (after merging within each side), renders
a row with two side-by-side 5x3 yes-rate heatmaps (Claude left, OpenAI
right). A scenario is included if EITHER:

  1) one or both models are internally contested (the 15 demographic
     variants don't unanimously agree on Yes vs No), OR
  2) the two models disagree on the overall majority verdict
     (e.g. Claude says "majority Yes" but OpenAI says "majority No").

Multiple files on a given side are concatenated row-wise before
yes-rates are computed, so an explain-no run (wider scenario coverage)
and an explain-yes run (has explanations) can be combined to maximise
both heatmap coverage and the set of cells with available explanations.

At the top, sticky sort-buttons reorder the rows client-side via JS:

  - Scenario ID (default, ascending)
  - Most similar models first  (mean |claude_yr - openai_yr| asc)
  - Most dissimilar models first (same, desc)
  - Claude internal disagreement  (Claude's spread desc)
  - OpenAI internal disagreement (OpenAI's spread desc)

Per-row metrics:
  similarity     = 1 - mean(|claude_yr - openai_yr|) over cells covered
                   in BOTH models. 1.00 = identical numbers, 0.00 = max diff.
  claude spread  = max - min Claude yes-rate over its covered cells.
  openai spread  = max - min OpenAI yes-rate over its covered cells.
  verdict pills  = each model's majority over its covered cells.

Missing cells (no Yes/No replicate on that side) render as a gray "—"
tile in the respective heatmap, so partial coverage stays visible.

Usage:
    python cross_model_gallery.py \\
        --claude <csv> [--claude <csv> ...] \\
        --openai <csv> [--openai <csv> ...] \\
        [--output reports/cross_model.html]

Default output: reports/cross_model_<claude_model>_vs_<openai_model>.html.
"""

import argparse
import csv
import html as _html
import math
import sys
from collections import defaultdict
from pathlib import Path

from variant_trends_report import (
    RACES, INCOMES, VARIANTS, majority,
    _sanitize_for_filename, resolve_non_clobbering,
)
from contested_gallery_report import (
    coverage_of, is_unanimous_partial, spread_of,
    yr_color, text_color, render_coverage_badge,
)


def esc(v) -> str:
    return _html.escape(str(v) if v is not None else "")


# ---------------------------------------------------------------------------
# Multi-file yes-rate loader
# ---------------------------------------------------------------------------

def load_partial_yes_rates_multi(paths: list[Path]) -> tuple[list[str], dict]:
    """Same shape as contested_gallery_report.load_partial_yes_rates but
    accepts a list of CSVs. Yes/No replicates from every file are pooled
    per (scenario_id, race, income) before the yes-rate ratio is computed,
    so coverage stacks: a scenario with no Yes/No in file A but data in
    file B becomes covered."""
    raw: dict = defaultdict(lambda: defaultdict(list))
    all_sids: set[str] = set()
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
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


# ---------------------------------------------------------------------------
# Per-scenario metrics
# ---------------------------------------------------------------------------

def mean_abs_diff(rates_a: dict, rates_b: dict, sid: str) -> float:
    """Mean |a_yr - b_yr| over cells with data on BOTH sides. NaN if no overlap."""
    diffs = []
    for v in VARIANTS:
        if sid in rates_a[v] and sid in rates_b[v]:
            diffs.append(abs(rates_a[v][sid] - rates_b[v][sid]))
    if not diffs:
        return float("nan")
    return sum(diffs) / len(diffs)


def model_verdict(rates: dict, sid: str) -> str:
    """Majority across covered cells. Returns 'Yes', 'No', or 'split'."""
    n_yes = n_no = 0
    for v in VARIANTS:
        if sid not in rates[v]:
            continue
        m = majority(rates[v][sid])
        if m == "Yes":
            n_yes += 1
        elif m == "No":
            n_no += 1
    if n_yes > n_no:
        return "Yes"
    if n_no > n_yes:
        return "No"
    return "split"


# ---------------------------------------------------------------------------
# SVG heatmap (no click handlers — this report is just visual comparison)
# ---------------------------------------------------------------------------

def heatmap_svg(rates: dict, sid: str, w: int = 300, h: int = 140) -> str:
    """Each cell is a clickable <g> with data-race / data-income that filters
    BOTH models' explanation tables (and highlights the matching cell on the
    other heatmap in the same row)."""
    cw = w / 5
    ch = h / 3
    parts = [f'<svg viewBox="-2 -18 {w + 60} {h + 24}" '
             f'xmlns="http://www.w3.org/2000/svg" '
             f'font-family="sans-serif" font-size="10">']
    for ri, race in enumerate(RACES):
        cx = ri * cw + cw / 2
        parts.append(
            f'<text x="{cx:.1f}" y="-4" text-anchor="middle" '
            f'font-size="10" fill="#444">{esc(race)}</text>'
        )
    for ii, inc in enumerate(INCOMES):
        for ri, race in enumerate(RACES):
            x, y = ri * cw, ii * ch
            v = (race, inc)
            if sid in rates[v]:
                yr = rates[v][sid]
                parts.append(
                    f'<g class="cell-click" data-race="{esc(race)}" '
                    f'data-income="{esc(inc)}" onclick="filterRow(this)">'
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw:.1f}" '
                    f'height="{ch:.1f}" fill="{yr_color(yr)}" '
                    f'stroke="#fff" stroke-width="1"/>'
                    f'<text x="{x + cw / 2:.1f}" y="{y + ch / 2 + 4:.1f}" '
                    f'text-anchor="middle" fill="{text_color(yr)}" '
                    f'font-size="11" font-variant-numeric="tabular-nums" '
                    f'pointer-events="none">{yr:.2f}</text>'
                    f'</g>'
                )
            else:
                parts.append(
                    f'<g class="cell-click cell-missing" data-race="{esc(race)}" '
                    f'data-income="{esc(inc)}" onclick="filterRow(this)">'
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw:.1f}" '
                    f'height="{ch:.1f}" fill="#ececec" stroke="#fff" '
                    f'stroke-width="1"/>'
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
            f'text-anchor="start" font-size="11" fill="#444">'
            f'{esc(short)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Paired explanations block (one details element per scenario row,
# containing two columns of explanations — Claude on the left, OpenAI on
# the right. Click handler filters both columns simultaneously.)
# ---------------------------------------------------------------------------

def _expl_sort_key(t):
    race, inc, rep, _, _ = t
    race_idx = {r: i for i, r in enumerate(RACES)}
    inc_idx = {i: k for k, i in enumerate(INCOMES)}
    try:
        rep_n = int(rep)
    except (TypeError, ValueError):
        rep_n = -1
    return (race_idx.get(race, 99), inc_idx.get(inc, 99), rep_n)


def render_expl_table(explanations: list[tuple]) -> str:
    """One model's per-replicate explanation table. Empty rendering if the
    list is empty (the wrapper handles the "no explanations" message)."""
    if not explanations:
        return ('<div class="no-expl">No explanations from this model for '
                'this scenario.</div>')
    rows_html: list[str] = []
    for race, inc, rep, answer, exp in sorted(explanations, key=_expl_sort_key):
        if answer == "Yes":
            ans_cls = "ans-yes"
        elif answer == "No":
            ans_cls = "ans-no"
        else:
            ans_cls = "ans-other"
        rows_html.append(
            f'<tr class="meta-row" data-race="{esc(race)}" '
            f'data-income="{esc(inc)}">'
            f'<td class="r-race">{esc(race)}</td>'
            f'<td class="r-inc">{esc(inc.replace(" income", ""))}</td>'
            f'<td class="r-rep">{esc(rep)}</td>'
            f'<td class="r-ans {ans_cls}">{esc(answer or "—")}</td>'
            f'</tr>'
            f'<tr class="exp-row" data-race="{esc(race)}" '
            f'data-income="{esc(inc)}">'
            f'<td colspan="4" class="r-exp">{esc(exp)}</td>'
            f'</tr>'
        )
    return (
        '<table class="expl-tbl"><thead><tr>'
        '<th>race</th><th>income</th><th>#</th><th>answer</th>'
        '</tr></thead><tbody>'
        + "".join(rows_html) +
        '</tbody></table>'
    )


def render_paired_explanations(claude_expls: list[tuple],
                               openai_expls: list[tuple]) -> str:
    """Two-column details block. Returns "" if BOTH sides are empty (no
    point in rendering an empty disclosure)."""
    n_c = len(claude_expls)
    n_o = len(openai_expls)
    if n_c == 0 and n_o == 0:
        return ""
    summary_html = (f"Show explanations &mdash; "
                    f"Claude: {n_c}, OpenAI: {n_o}")
    return (
        '<details class="expl">'
        f'<summary>{summary_html}</summary>'
        '<div class="expl-pair">'
        '<div class="expl-side" data-side="claude">'
        '<h4>Claude</h4>'
        + render_expl_table(claude_expls) +
        '</div>'
        '<div class="expl-side" data-side="openai">'
        '<h4>OpenAI</h4>'
        + render_expl_table(openai_expls) +
        '</div>'
        '</div></details>'
    )


# ---------------------------------------------------------------------------
# Row rendering
# ---------------------------------------------------------------------------

def render_verdict_pill(verdict: str, side: str) -> str:
    if verdict == "Yes":
        cls, label = "verdict-yes", "majority Yes"
    elif verdict == "No":
        cls, label = "verdict-no", "majority No"
    else:
        cls, label = "verdict-split", "split / no majority"
    return f'<span class="verdict {cls}">{esc(side)}: {label}</span>'


def render_row(sid: str,
               claude_rates: dict, openai_rates: dict,
               orig: dict, sim_score: float,
               claude_spread: float, openai_spread: float,
               claude_verdict: str, openai_verdict: str,
               claude_covg: int, openai_covg: int,
               claude_expls: list[tuple],
               openai_expls: list[tuple]) -> str:
    cross_disagree = (
        claude_verdict in ("Yes", "No") and openai_verdict in ("Yes", "No")
        and claude_verdict != openai_verdict
    )
    cross_badge = ('<span class="cross-disagree">&#x26A0; cross-model disagreement</span>'
                   if cross_disagree else
                   '<span class="cross-agree">models agree on verdict</span>')
    text = orig.get(sid) or "(no original scenario text)"
    sim_display = "n/a" if math.isnan(sim_score) else f"{(1 - sim_score):.2f}"
    # For sort: NaN sims get pushed to the "most dissimilar" end (1.0).
    sim_data = "1.0" if math.isnan(sim_score) else f"{sim_score:.4f}"
    return (
        f'<div class="row" '
        f'data-sid="{esc(sid)}" '
        f'data-similarity="{sim_data}" '
        f'data-claude-spread="{claude_spread:.4f}" '
        f'data-openai-spread="{openai_spread:.4f}">'
        f'<div class="row-header">'
        f'<strong class="sid">scenario {esc(sid)}</strong>'
        f'<span class="sep">&middot;</span>'
        f'<span class="metric">similarity <strong>{sim_display}</strong></span>'
        f'<span class="sep">&middot;</span>'
        f'{render_verdict_pill(claude_verdict, "Claude")} '
        f'{render_verdict_pill(openai_verdict, "OpenAI")}'
        f'<span class="sep">&middot;</span>'
        f'{cross_badge}'
        f'</div>'
        f'<div class="orig">{esc(text)}</div>'
        f'<div class="pair">'
        f'<div class="model-tile">'
        f'<div class="model-head"><strong>Claude</strong> '
        f'<span class="metric-inline">spread {claude_spread:.2f}</span> '
        f'{render_coverage_badge(claude_covg)}</div>'
        f'{heatmap_svg(claude_rates, sid)}'
        f'</div>'
        f'<div class="model-tile">'
        f'<div class="model-head"><strong>OpenAI</strong> '
        f'<span class="metric-inline">spread {openai_spread:.2f}</span> '
        f'{render_coverage_badge(openai_covg)}</div>'
        f'{heatmap_svg(openai_rates, sid)}'
        f'</div>'
        f'</div>'
        f'{render_paired_explanations(claude_expls, openai_expls)}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Color key + page chrome
# ---------------------------------------------------------------------------

def render_color_key(steps: int = 11, swatch_w: int = 36,
                     swatch_h: int = 24) -> str:
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
        '<span><strong>0.00</strong> &mdash; every replicate '
        '<strong style="color:#b13030">No</strong></span>'
        '<span><strong>0.50</strong> &mdash; split / ambiguous</span>'
        '<span><strong>1.00</strong> &mdash; every replicate '
        '<strong style="color:#1a4a8a">Yes</strong></span>'
        '<span><span class="swatch" style="background:#ececec; color:#888;'
        ' width:28px; height:16px; display:inline-flex; align-items:center;'
        ' justify-content:center;">&mdash;</span> missing cell '
        '(no Yes/No replicate on that side)</span>'
        '</div>'
    )
    parts.append('</div>')
    return "".join(parts)


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1400px; margin: 1em auto; padding: 0 1em; color: #222; }
h1 { font-size: 1.55em; margin-bottom: 0.2em; }
.meta { color: #666; font-size: 0.9em; margin-bottom: 1em; }
.meta code { background: #f0f0f0; padding: 0 4px; border-radius: 3px; }

.key { background: #f7f7f7; border: 1px solid #ddd; border-radius: 4px;
       padding: 10px 14px; margin: 0.5em 0 1em 0; }
.key-title { font-size: 0.92em; margin-bottom: 6px; color: #333; }
.key-title em { color: #555; font-style: normal; font-weight: 500; }
.swatch-row { display: flex; gap: 0; }
.swatch { display: flex; align-items: center; justify-content: center;
          font-size: 0.78em; font-variant-numeric: tabular-nums;
          border-right: 1px solid white; }
.swatch:first-child { border-radius: 3px 0 0 3px; }
.swatch:last-child  { border-radius: 0 3px 3px 0; border-right: none; }
.key-legend { display: flex; flex-wrap: wrap; gap: 1.5em; margin-top: 8px;
              font-size: 0.85em; color: #555; align-items: center; }
.key-legend strong { color: #222; }

.controls { position: sticky; top: 0; background: white; z-index: 10;
            border-bottom: 1px solid #ddd; padding: 10px 0;
            margin-bottom: 1em; display: flex; flex-wrap: wrap; gap: 8px;
            align-items: center; }
.controls .label { font-size: 0.9em; color: #555; margin-right: 6px;
                   font-weight: 600; }
.controls button { background: white; border: 1px solid #aaa; padding: 5px 12px;
                   border-radius: 12px; font-size: 0.92em; cursor: pointer;
                   font-family: inherit; color: #225; }
.controls button:hover { background: #eef; border-color: #225; }
.controls button.active { background: #225; color: white; border-color: #225; }

.gallery { display: flex; flex-direction: column; gap: 14px; }
.row { border: 1px solid #ddd; padding: 10px 14px; background: #fcfcfc;
       border-radius: 4px; }
.row-header { font-size: 0.9em; color: #333; margin-bottom: 6px;
              display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
.row-header .sid { font-size: 1.05em; color: #111; }
.row-header .sep { color: #bbb; }
.row-header .metric { color: #444; }
.row-header .metric strong { color: #111;
                              font-variant-numeric: tabular-nums; }

.verdict { display: inline-block; padding: 1px 8px; border-radius: 9px;
           font-size: 0.85em; }
.verdict-yes { background: hsl(220, 60%, 92%); color: #1a4a8a; }
.verdict-no  { background: hsl(0, 60%, 92%); color: #b13030; }
.verdict-split { background: #f0f0f0; color: #666; font-style: italic; }
.cross-disagree { background: hsl(45, 90%, 85%); color: #8a4a00;
                  font-weight: 600; padding: 1px 8px; border-radius: 9px;
                  font-size: 0.85em; }
.cross-agree { color: #888; font-size: 0.85em; }

.orig { font-size: 0.92em; line-height: 1.5; color: #222;
        margin: 0 0 10px 0; white-space: pre-wrap; }

.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.model-tile { padding: 8px; background: #fff; border: 1px solid #e3e3e3;
              border-radius: 4px; }
.model-head { font-size: 0.88em; color: #333; margin-bottom: 6px;
              display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
.model-tile svg { width: 100%; height: auto; }
.metric-inline { color: #555; font-size: 0.9em;
                 font-variant-numeric: tabular-nums; }

.cov-badge { display: inline-block; padding: 1px 8px; border-radius: 9px;
             font-size: 0.82em; white-space: nowrap; }
.cov-badge.cov-full { background: hsl(120, 50%, 92%); color: #2a6b2a; }
.cov-badge.cov-partial { background: hsl(45, 75%, 88%); color: #8a6a00;
                         font-weight: 500; }
.cov-badge.cov-low { background: hsl(0, 70%, 92%); color: #b13030;
                     font-weight: 600; }

/* Clickable heatmap cells */
g.cell-click { cursor: pointer; }
g.cell-click:hover rect { stroke: #444; stroke-width: 2; }
g.cell-click.active rect { stroke: #111; stroke-width: 3; }
g.cell-missing:hover rect { stroke: #888; }
g.cell-missing.active rect { stroke: #444; }

/* Paired explanations block */
details.expl { margin-top: 10px; border-top: 1px solid #eee; padding-top: 8px; }
details.expl > summary { cursor: pointer; font-size: 0.9em; color: #225;
                         padding: 4px 0; user-select: none; }
details.expl > summary:hover { color: #003; }
details.expl[open] > summary { font-weight: 600; }
.expl-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
             margin-top: 8px; }
.expl-side { background: #fafafa; border: 1px solid #e3e3e3;
             border-radius: 4px; padding: 8px 10px; min-width: 0; }
.expl-side h4 { margin: 0 0 6px 0; font-size: 0.95em; color: #333; }
.expl-side .no-expl { color: #888; font-size: 0.9em; font-style: italic;
                      padding: 6px 2px; }

table.expl-tbl { width: 100%; border-collapse: collapse; font-size: 0.85em; }
table.expl-tbl th { background: #f0f0f0; text-align: left; font-weight: 600;
                    color: #333; padding: 3px 6px; border: 1px solid #e3e3e3; }
table.expl-tbl tr.meta-row > td { padding: 3px 6px; vertical-align: middle;
                                   border-left: 1px solid #e3e3e3;
                                   border-right: 1px solid #e3e3e3;
                                   border-top: 1px solid #e3e3e3; }
table.expl-tbl td.r-race { width: 28%; color: #444; }
table.expl-tbl td.r-inc  { width: 22%; color: #444; }
table.expl-tbl td.r-rep  { width: 10%; text-align: right;
                            font-variant-numeric: tabular-nums; color: #888; }
table.expl-tbl td.r-ans  { width: 22%; font-weight: 600; text-align: center; }
table.expl-tbl td.ans-yes { color: #1a8830; background: hsl(220, 60%, 96%); }
table.expl-tbl td.ans-no  { color: #b13030; background: hsl(0, 60%, 96%); }
table.expl-tbl td.ans-other { color: #888; font-style: italic; }
table.expl-tbl tr.exp-row > td.r-exp {
    background: #fff; border: 1px solid #e3e3e3; border-top: none;
    padding: 5px 8px 8px 8px; line-height: 1.45; color: #333;
    white-space: pre-wrap; word-wrap: break-word;
    border-bottom: 2px solid #e0e0e0; font-size: 0.97em; }

button.clear-link { background: white; border: 1px solid #aaa;
                    padding: 1px 8px; border-radius: 9px; font-size: 0.88em;
                    color: #225; cursor: pointer; font-family: inherit;
                    margin-left: 4px; }
button.clear-link:hover { background: #eef; border-color: #225; }
"""

JS = """
function sortGallery(metric, dir, btn) {
  const g = document.querySelector('.gallery');
  const rows = Array.from(g.querySelectorAll('.row'));
  rows.sort(function(a, b) {
    const va = a.dataset[metric];
    const vb = b.dataset[metric];
    const na = parseFloat(va);
    const nb = parseFloat(vb);
    let cmp;
    if (!isNaN(na) && !isNaN(nb)) {
      cmp = na - nb;
    } else {
      cmp = String(va).localeCompare(String(vb));
    }
    return dir * cmp;
  });
  // Re-append to gallery in sorted order. Using a DocumentFragment keeps
  // the reflow count to one even with hundreds of rows.
  const frag = document.createDocumentFragment();
  for (const row of rows) frag.appendChild(row);
  g.appendChild(frag);
  document.querySelectorAll('.controls button').forEach(function(b) {
    b.classList.remove('active');
  });
  if (btn) btn.classList.add('active');
}

function filterRow(g) {
  // Click handler on a cell <g>. Opens the row's <details> and filters
  // BOTH side-by-side tables to that (race, income) group. Also highlights
  // the SAME (race, income) cell on the OTHER model's heatmap so the user
  // can see which cells the parallel explanations correspond to.
  const race = g.dataset.race;
  const income = g.dataset.income;
  const row = g.closest('.row');
  const det = row.querySelector('details.expl');
  if (!det) return;

  const filterKey = race + '|' + income;
  if (det.dataset.filter === filterKey) {
    // Same cell twice — collapse + clear, matching the contested_gallery UX.
    clearRowFilter(row);
    det.open = false;
    return;
  }

  det.open = true;
  det.dataset.filter = filterKey;

  const summary = det.querySelector('summary');
  if (!summary.dataset.allLabel) summary.dataset.allLabel = summary.innerHTML;

  const counts = {claude: {visible: 0, total: 0},
                  openai: {visible: 0, total: 0}};
  for (const side of det.querySelectorAll('.expl-side')) {
    const sideKey = side.dataset.side;
    const tbody = side.querySelector('table.expl-tbl tbody');
    if (!tbody) continue;
    for (const tr of tbody.rows) {
      const isMeta = tr.classList.contains('meta-row');
      if (isMeta) counts[sideKey].total++;
      if (tr.dataset.race === race && tr.dataset.income === income) {
        tr.style.display = '';
        if (isMeta) counts[sideKey].visible++;
      } else {
        tr.style.display = 'none';
      }
    }
  }

  // Highlight the (race, income) cell on BOTH heatmaps in this row.
  for (const el of row.querySelectorAll('.cell-click.active')) {
    el.classList.remove('active');
  }
  const sel = '.cell-click[data-race="' + race + '"][data-income="' + income + '"]';
  for (const cell of row.querySelectorAll(sel)) {
    cell.classList.add('active');
  }

  const incShort = income.replace(' income', '');
  summary.innerHTML = 'Showing <strong>' + race + ' / ' + incShort
    + '</strong> &mdash; Claude ' + counts.claude.visible + '/'
    + counts.claude.total + ', OpenAI ' + counts.openai.visible + '/'
    + counts.openai.total + ' explanations '
    + '<button type="button" class="clear-link" '
    + 'onclick="event.preventDefault(); event.stopPropagation(); '
    + 'clearRowFilter(this.closest(&quot;.row&quot;));">clear filter</button>';
}

function clearRowFilter(row) {
  const det = row.querySelector('details.expl');
  if (!det) return;
  det.dataset.filter = '';
  for (const tbody of det.querySelectorAll('table.expl-tbl tbody')) {
    for (const tr of tbody.rows) tr.style.display = '';
  }
  const summary = det.querySelector('summary');
  if (summary.dataset.allLabel) summary.innerHTML = summary.dataset.allLabel;
  for (const el of row.querySelectorAll('.cell-click.active')) {
    el.classList.remove('active');
  }
}
"""


def _format_path_list(paths: list[Path]) -> str:
    return "<br>".join(f"&nbsp;&nbsp;<code>{esc(p)}</code>" for p in paths)


def build_html(claude_paths: list[Path], openai_paths: list[Path],
               claude_model: str, openai_model: str,
               n_both: int, n_shown: int,
               n_claude_only: int, n_openai_only: int,
               rows_html: str) -> str:
    c_label = f"{len(claude_paths)} file" + ("s" if len(claude_paths) != 1 else "")
    o_label = f"{len(openai_paths)} file" + ("s" if len(openai_paths) != 1 else "")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Cross-model contested-scenario gallery</title>
<style>{CSS}</style>
<script>{JS}</script>
</head><body>
<h1>Cross-model contested-scenario gallery</h1>
<div class="meta">
Claude: <code>{esc(claude_model)}</code> &mdash; {c_label}:<br>
{_format_path_list(claude_paths)}<br>
OpenAI: <code>{esc(openai_model)}</code> &mdash; {o_label}:<br>
{_format_path_list(openai_paths)}<br>
<strong>{n_shown}</strong> scenarios shown &mdash; included when at least
one model is internally contested OR the two models disagree on the
majority verdict.
Source overlap: {n_both} scenarios in both sides
(claude-only: {n_claude_only}, openai-only: {n_openai_only}).
</div>

{render_color_key()}

<div class="controls">
<span class="label">Sort:</span>
<button class="active" onclick="sortGallery('sid', 1, this)">Scenario ID &uarr;</button>
<button onclick="sortGallery('similarity', 1, this)">Most similar models</button>
<button onclick="sortGallery('similarity', -1, this)">Most dissimilar models</button>
<button onclick="sortGallery('claudeSpread', -1, this)">Claude internal disagreement</button>
<button onclick="sortGallery('openaiSpread', -1, this)">OpenAI internal disagreement</button>
</div>

<div class="gallery">{rows_html}</div>
</body></html>
"""


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--claude", action="append", default=[], metavar="CSV",
                   help="Claude results CSV. Repeat to merge multiple files "
                        "(e.g. an explain-no run plus an explain-yes run).")
    p.add_argument("--openai", action="append", default=[], metavar="CSV",
                   help="OpenAI results CSV. Repeat to merge multiple files.")
    p.add_argument("--output", default=None,
                   help="Output HTML path. Default: "
                        "reports/cross_model_<claude>_vs_<openai>.html.")
    args = p.parse_args()

    if not args.claude or not args.openai:
        p.error("at least one --claude and one --openai CSV are required")

    claude_paths = [Path(c) for c in args.claude]
    openai_paths = [Path(o) for o in args.openai]
    for p_ in (*claude_paths, *openai_paths):
        if not p_.exists():
            print(f"error: not found: {p_}", file=sys.stderr)
            sys.exit(2)

    claude_sids, claude_rates = load_partial_yes_rates_multi(claude_paths)
    openai_sids, openai_rates = load_partial_yes_rates_multi(openai_paths)
    claude_set = set(claude_sids)
    openai_set = set(openai_sids)
    both_set = claude_set & openai_set

    claude_only = len(claude_set - openai_set)
    openai_only = len(openai_set - claude_set)

    # Compute per-scenario metrics for the intersection only.
    metrics = {}
    for s in both_set:
        metrics[s] = {
            "claude_covg":    coverage_of(claude_rates, s),
            "openai_covg":    coverage_of(openai_rates, s),
            "claude_spread":  spread_of(claude_rates, s),
            "openai_spread":  spread_of(openai_rates, s),
            "claude_verdict": model_verdict(claude_rates, s),
            "openai_verdict": model_verdict(openai_rates, s),
            "sim":            mean_abs_diff(claude_rates, openai_rates, s),
        }

    # Filter: include if either model is contested OR cross-model verdicts diverge.
    candidates = []
    for s in sorted(both_set):
        m = metrics[s]
        c_contested = not is_unanimous_partial(claude_rates, s)
        o_contested = not is_unanimous_partial(openai_rates, s)
        cross = (m["claude_verdict"] in ("Yes", "No")
                 and m["openai_verdict"] in ("Yes", "No")
                 and m["claude_verdict"] != m["openai_verdict"])
        if c_contested or o_contested or cross:
            candidates.append(s)

    if not candidates:
        print("error: no scenarios meet the filter criteria.", file=sys.stderr)
        sys.exit(2)

    # Pull original scenario text + explanations from every input CSV, one
    # pass per file. Explanations from any file (typically the explain-yes
    # run) attach to whichever side the file came from.
    orig: dict[str, str] = {}
    claude_expls: dict[str, list[tuple]] = {}
    openai_expls: dict[str, list[tuple]] = {}
    candidates_set = set(candidates)

    def collect(path, expls_out, model_box):
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                sid = r["scenario_id"]
                if not model_box["model"] and r.get("model"):
                    model_box["model"] = r["model"]
                if sid not in candidates_set:
                    continue
                if sid not in orig:
                    t = r.get("original_scenario", "")
                    if t:
                        orig[sid] = t
                exp = (r.get("explanation") or "").strip()
                if not exp:
                    continue
                expls_out.setdefault(sid, []).append((
                    r.get("race_variant", ""),
                    r.get("income_variant", ""),
                    r.get("replicate_idx", ""),
                    r.get("answer", ""),
                    exp,
                ))

    c_box = {"model": ""}
    o_box = {"model": ""}
    for cp in claude_paths:
        collect(cp, claude_expls, c_box)
    for op in openai_paths:
        collect(op, openai_expls, o_box)
    claude_model = c_box["model"]
    openai_model = o_box["model"]

    rows_html = "".join(
        render_row(s, claude_rates, openai_rates, orig,
                   metrics[s]["sim"],
                   metrics[s]["claude_spread"], metrics[s]["openai_spread"],
                   metrics[s]["claude_verdict"], metrics[s]["openai_verdict"],
                   metrics[s]["claude_covg"], metrics[s]["openai_covg"],
                   claude_expls.get(s, []),
                   openai_expls.get(s, []))
        for s in candidates
    )

    if args.output:
        out = Path(args.output)
    else:
        c_label = _sanitize_for_filename(claude_model or "claude")
        o_label = _sanitize_for_filename(openai_model or "openai")
        out = (Path(__file__).resolve().parent / "reports"
               / f"cross_model_{c_label}_vs_{o_label}.html")
    output_path = resolve_non_clobbering(out)
    if output_path != out:
        print(f"warning: {out} exists; writing to {output_path}",
              file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_html(claude_paths, openai_paths, claude_model, openai_model,
                   len(both_set), len(candidates),
                   claude_only, openai_only, rows_html),
        encoding="utf-8",
    )
    print(f"wrote {len(candidates)} scenarios to {output_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
