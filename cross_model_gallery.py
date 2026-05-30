#!/usr/bin/env python3
"""Cross-model contested-scenario gallery.

Takes one Claude CSV and one OpenAI CSV. For each scenario present in BOTH
CSVs, renders a row with two side-by-side 5x3 yes-rate heatmaps (Claude
left, OpenAI right). A scenario is included if EITHER:

  1) one or both models are internally contested (the 15 demographic
     variants don't unanimously agree on Yes vs No), OR
  2) the two models disagree on the overall majority verdict
     (e.g. Claude says "majority Yes" but OpenAI says "majority No").

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
    python cross_model_gallery.py <claude_csv> <openai_csv>
        [--output reports/cross_model.html]

Default output: reports/cross_model_<claude_model>_vs_<openai_model>.html.
"""

import argparse
import csv
import html as _html
import math
import sys
from pathlib import Path

from variant_trends_report import (
    RACES, INCOMES, VARIANTS, majority,
    _sanitize_for_filename, resolve_non_clobbering,
)
from contested_gallery_report import (
    load_partial_yes_rates, coverage_of, is_unanimous_partial, spread_of,
    yr_color, text_color, render_coverage_badge,
)


def esc(v) -> str:
    return _html.escape(str(v) if v is not None else "")


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
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw:.1f}" '
                    f'height="{ch:.1f}" fill="{yr_color(yr)}" '
                    f'stroke="#fff" stroke-width="1"/>'
                    f'<text x="{x + cw / 2:.1f}" y="{y + ch / 2 + 4:.1f}" '
                    f'text-anchor="middle" fill="{text_color(yr)}" '
                    f'font-size="11" font-variant-numeric="tabular-nums">'
                    f'{yr:.2f}</text>'
                )
            else:
                parts.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw:.1f}" '
                    f'height="{ch:.1f}" fill="#ececec" stroke="#fff" '
                    f'stroke-width="1"/>'
                    f'<line x1="{x + 2:.1f}" y1="{y + 2:.1f}" '
                    f'x2="{x + cw - 2:.1f}" y2="{y + ch - 2:.1f}" '
                    f'stroke="#cfcfcf" stroke-width="1"/>'
                    f'<text x="{x + cw / 2:.1f}" y="{y + ch / 2 + 4:.1f}" '
                    f'text-anchor="middle" fill="#888" '
                    f'font-size="11">&#8212;</text>'
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
               claude_covg: int, openai_covg: int) -> str:
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
"""


def build_html(claude_path: Path, openai_path: Path,
               claude_model: str, openai_model: str,
               n_both: int, n_shown: int,
               n_claude_only: int, n_openai_only: int,
               rows_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Cross-model contested-scenario gallery</title>
<style>{CSS}</style>
<script>{JS}</script>
</head><body>
<h1>Cross-model contested-scenario gallery</h1>
<div class="meta">
Claude: <code>{esc(claude_model)}</code> &mdash; <code>{esc(claude_path)}</code><br>
OpenAI: <code>{esc(openai_model)}</code> &mdash; <code>{esc(openai_path)}</code><br>
<strong>{n_shown}</strong> scenarios shown &mdash; included when at least
one model is internally contested OR the two models disagree on the
majority verdict.
Source overlap: {n_both} scenarios in both CSVs
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
    p.add_argument("claude_csv", help="Path to a Claude results CSV.")
    p.add_argument("openai_csv", help="Path to an OpenAI results CSV.")
    p.add_argument("--output", default=None,
                   help="Output HTML path. Default: "
                        "reports/cross_model_<claude>_vs_<openai>.html.")
    args = p.parse_args()

    claude_path = Path(args.claude_csv)
    openai_path = Path(args.openai_csv)
    for p_ in (claude_path, openai_path):
        if not p_.exists():
            print(f"error: not found: {p_}", file=sys.stderr)
            sys.exit(2)

    claude_sids, claude_rates = load_partial_yes_rates(claude_path)
    openai_sids, openai_rates = load_partial_yes_rates(openai_path)
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

    # Pull original scenario text from Claude first, OpenAI as fallback.
    orig: dict[str, str] = {}
    claude_model = openai_model = ""
    candidates_set = set(candidates)
    with claude_path.open(newline="") as f:
        for r in csv.DictReader(f):
            sid = r["scenario_id"]
            if not claude_model and r.get("model"):
                claude_model = r["model"]
            if sid in candidates_set and sid not in orig:
                t = r.get("original_scenario", "")
                if t:
                    orig[sid] = t
    missing = candidates_set - set(orig)
    with openai_path.open(newline="") as f:
        for r in csv.DictReader(f):
            sid = r["scenario_id"]
            if not openai_model and r.get("model"):
                openai_model = r["model"]
            if sid in missing and sid not in orig:
                t = r.get("original_scenario", "")
                if t:
                    orig[sid] = t

    rows_html = "".join(
        render_row(s, claude_rates, openai_rates, orig,
                   metrics[s]["sim"],
                   metrics[s]["claude_spread"], metrics[s]["openai_spread"],
                   metrics[s]["claude_verdict"], metrics[s]["openai_verdict"],
                   metrics[s]["claude_covg"], metrics[s]["openai_covg"])
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
        build_html(claude_path, openai_path, claude_model, openai_model,
                   len(both_set), len(candidates),
                   claude_only, openai_only, rows_html),
        encoding="utf-8",
    )
    print(f"wrote {len(candidates)} scenarios to {output_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
