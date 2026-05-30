#!/usr/bin/env python3
"""Consensus summary report — bar charts.

Two visualizations, computed over the scenario intersection of the two
sides (the only set where a cross-model comparison is meaningful):

  1) Per-model internal consensus: for each model, the # of intersection
     scenarios where every covered cell agrees on a Yes-or-No majority
     ("consensus") vs scenarios where at least one cell disagrees with
     another ("non-consensus"). Same `is_unanimous_partial` check the
     contested gallery uses, applied independently per model.

  2) Cross-model agreement on consensus scenarios: for each model's
     internally-consensus scenarios, the # where the OTHER model's
     overall verdict (majority over its covered cells) agrees with
     this model's consensus verdict, disagrees with it, or is split
     (the other model's covered cells produce a tie, so we can't call
     it agreement or disagreement).

Multiple files per side are pooled before yes-rates are computed, so an
explain-no and explain-yes run on the same model combine to maximise
coverage.

Usage:
    python consensus_summary_report.py \\
        --claude <csv> [--claude <csv> ...] \\
        --openai <csv> [--openai <csv> ...] \\
        [--output reports/consensus_summary.html]

Default output: reports/consensus_summary_<claude>_vs_<openai>.html.
"""

import argparse
import csv
import html as _html
import sys
from pathlib import Path

from variant_trends_report import (
    _sanitize_for_filename, resolve_non_clobbering,
)
from contested_gallery_report import is_unanimous_partial
from cross_model_gallery import load_partial_yes_rates_multi, model_verdict


def esc(v) -> str:
    return _html.escape(str(v) if v is not None else "")


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

COL_CONSENSUS     = "#2a8a3e"   # green
COL_NONCONSENSUS  = "#d68910"   # amber
COL_AGREE         = "#1a4a8a"   # blue
COL_DISAGREE      = "#b13030"   # red
COL_NOVERDICT     = "#888888"   # gray
COL_TEXT_ON_DARK  = "#ffffff"


# ---------------------------------------------------------------------------
# SVG primitives
# ---------------------------------------------------------------------------

def stacked_bar(segments, total, width=720, height=42) -> str:
    """Horizontal stacked bar. segments = [(label, count, fill), ...].
    Renders "{label}: {count} ({pct}%)" inside wide-enough segments, just the
    count inside narrower ones, nothing in razor-thin ones. Returns "" if
    total <= 0."""
    if total <= 0:
        return ('<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
                'font-family="sans-serif" font-size="13">'
                '<rect width="{w}" height="{h}" fill="#f3f3f3" '
                'stroke="#ddd"/><text x="{tx}" y="{ty}" text-anchor="middle" '
                'fill="#888">no data</text></svg>').format(
                    w=width, h=height, tx=width/2, ty=height/2 + 4)
    parts = [f'<svg viewBox="0 0 {width} {height}" '
             f'xmlns="http://www.w3.org/2000/svg" '
             f'font-family="sans-serif" font-size="13">']
    x = 0.0
    for seg_label, count, color in segments:
        if count <= 0:
            continue
        seg_w = count / total * width
        if seg_w < 0.5:
            continue
        parts.append(
            f'<rect x="{x:.2f}" y="0" width="{seg_w:.2f}" height="{height}" '
            f'fill="{color}"/>'
        )
        pct = (count / total) * 100
        text_full = f"{seg_label}: {count} ({pct:.0f}%)"
        if seg_w >= 130:
            parts.append(
                f'<text x="{x + seg_w/2:.2f}" y="{height/2 + 4.5}" '
                f'text-anchor="middle" fill="{COL_TEXT_ON_DARK}" '
                f'font-weight="600">{esc(text_full)}</text>'
            )
        elif seg_w >= 26:
            parts.append(
                f'<text x="{x + seg_w/2:.2f}" y="{height/2 + 4.5}" '
                f'text-anchor="middle" fill="{COL_TEXT_ON_DARK}" '
                f'font-weight="600">{count}</text>'
            )
        x += seg_w
    parts.append('</svg>')
    return "".join(parts)


def bar_row(label: str, sublabel: str, segments, total: int) -> str:
    """Two-line label on the left + stacked bar + total on the right."""
    return (
        f'<div class="bar-row">'
        f'<div class="bar-label">'
        f'<div class="bar-label-main">{esc(label)}</div>'
        f'<div class="bar-label-sub">{esc(sublabel)}</div>'
        f'</div>'
        f'<div class="bar-bar">{stacked_bar(segments, total)}</div>'
        f'<div class="bar-total">{total}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def cross_compare(consensus_sids, my_rates, other_rates):
    """For each sid in consensus_sids, classify what the OTHER model's verdict
    says about it: agree / disagree / no_verdict (other side splits)."""
    agree = disagree = no_verdict = 0
    for sid in consensus_sids:
        my_v = model_verdict(my_rates, sid)
        other_v = model_verdict(other_rates, sid)
        if other_v == "split":
            no_verdict += 1
        elif other_v == my_v:
            agree += 1
        else:
            disagree += 1
    return agree, disagree, no_verdict


# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1100px; margin: 1em auto; padding: 0 1em; color: #222; }
h1 { font-size: 1.55em; margin-bottom: 0.2em; }
h2 { font-size: 1.15em; margin-top: 1.8em; margin-bottom: 0.4em;
     padding-bottom: 4px; border-bottom: 1px solid #ddd; }
.meta { color: #666; font-size: 0.9em; margin-bottom: 1em; }
.meta code { background: #f0f0f0; padding: 0 4px; border-radius: 3px; }
.section-note { color: #555; font-size: 0.92em; margin: 0 0 12px 0;
                line-height: 1.5; }

.bar-row { display: grid; grid-template-columns: 180px 1fr 60px;
           gap: 14px; align-items: center; margin: 8px 0; }
.bar-label-main { font-weight: 600; color: #222; font-size: 0.98em; }
.bar-label-sub  { color: #666; font-size: 0.82em; margin-top: 1px; }
.bar-bar svg { display: block; width: 100%; height: 42px; }
.bar-total { text-align: right; color: #444; font-weight: 600;
             font-variant-numeric: tabular-nums; }

.legend { display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
          margin: 6px 0 14px 0; font-size: 0.88em; color: #444; }
.legend .sw { display: inline-block; width: 14px; height: 14px;
              border-radius: 3px; vertical-align: middle;
              margin-right: 5px; }

table.summary { border-collapse: collapse; margin-top: 8px; font-size: 0.92em; }
table.summary th, table.summary td { padding: 4px 10px;
    border: 1px solid #e3e3e3; text-align: right;
    font-variant-numeric: tabular-nums; }
table.summary th { background: #f0f0f0; text-align: center; font-weight: 600; }
table.summary td.label-cell { text-align: left; font-weight: 500; }
"""


def render_legend(items) -> str:
    parts = ['<div class="legend">']
    for label, color in items:
        parts.append(f'<span><span class="sw" style="background:{color}"></span>'
                     f'{esc(label)}</span>')
    parts.append('</div>')
    return "".join(parts)


def render_summary_table(headers, rows) -> str:
    head = "".join(f'<th>{esc(h)}</th>' for h in headers)
    body_rows = []
    for r in rows:
        cells = []
        for i, val in enumerate(r):
            cls = "label-cell" if i == 0 else ""
            cells.append(f'<td class="{cls}">{esc(val)}</td>')
        body_rows.append('<tr>' + "".join(cells) + '</tr>')
    return ('<table class="summary"><thead><tr>' + head +
            '</tr></thead><tbody>' + "".join(body_rows) + '</tbody></table>')


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_html(claude_paths, openai_paths,
               claude_model, openai_model,
               n_intersection, n_claude_only, n_openai_only,
               c_consensus, c_non, c_verdict_breakdown,
               o_consensus, o_non, o_verdict_breakdown,
               cc_agree, cc_disagree, cc_no_verdict,
               oo_agree, oo_disagree, oo_no_verdict) -> str:
    c_total = c_consensus + c_non
    o_total = o_consensus + o_non

    section1_legend = render_legend([
        (f"consensus (every covered cell agrees on Yes-or-No majority)",
         COL_CONSENSUS),
        ("non-consensus (at least one cell's majority disagrees)",
         COL_NONCONSENSUS),
    ])
    section2_legend = render_legend([
        ("other model AGREES (its overall verdict matches)", COL_AGREE),
        ("other model DISAGREES (verdict flips)", COL_DISAGREE),
        ("other model SPLIT (no overall verdict to compare)", COL_NOVERDICT),
    ])

    bar1 = bar_row(
        "Claude", "internal consensus",
        [("consensus", c_consensus, COL_CONSENSUS),
         ("non-consensus", c_non, COL_NONCONSENSUS)],
        c_total)
    bar2 = bar_row(
        "OpenAI", "internal consensus",
        [("consensus", o_consensus, COL_CONSENSUS),
         ("non-consensus", o_non, COL_NONCONSENSUS)],
        o_total)

    cy, cn = c_verdict_breakdown
    oy, on = o_verdict_breakdown
    section1_summary = render_summary_table(
        ["model", "consensus", "  -> Yes", "  -> No",
         "non-consensus", "total"],
        [["Claude", c_consensus, cy, cn, c_non, c_total],
         ["OpenAI", o_consensus, oy, on, o_non, o_total]],
    )

    bar3 = bar_row(
        "Claude consensus", "→ what does OpenAI say?",
        [("OpenAI agrees", cc_agree, COL_AGREE),
         ("OpenAI disagrees", cc_disagree, COL_DISAGREE),
         ("OpenAI split", cc_no_verdict, COL_NOVERDICT)],
        c_consensus)
    bar4 = bar_row(
        "OpenAI consensus", "→ what does Claude say?",
        [("Claude agrees", oo_agree, COL_AGREE),
         ("Claude disagrees", oo_disagree, COL_DISAGREE),
         ("Claude split", oo_no_verdict, COL_NOVERDICT)],
        o_consensus)

    section2_summary = render_summary_table(
        ["perspective", "agree", "disagree", "other split", "total"],
        [["Claude consensus → OpenAI", cc_agree, cc_disagree, cc_no_verdict,
          c_consensus],
         ["OpenAI consensus → Claude", oo_agree, oo_disagree, oo_no_verdict,
          o_consensus]],
    )

    def fmt_paths(paths):
        return "<br>".join(f"&nbsp;&nbsp;<code>{esc(p)}</code>" for p in paths)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Consensus summary &mdash; {esc(claude_model)} vs {esc(openai_model)}</title>
<style>{CSS}</style>
</head><body>

<h1>Consensus summary</h1>
<div class="meta">
Claude: <code>{esc(claude_model)}</code> &mdash; {len(claude_paths)} file(s):<br>
{fmt_paths(claude_paths)}<br>
OpenAI: <code>{esc(openai_model)}</code> &mdash; {len(openai_paths)} file(s):<br>
{fmt_paths(openai_paths)}<br>
Intersection: <strong>{n_intersection}</strong> scenarios in both sides
(claude-only: {n_claude_only}, openai-only: {n_openai_only}). All counts
below are over the intersection.
</div>

<h2>1. Per-model internal consensus</h2>
<p class="section-note">
For each model, how many of the {n_intersection} intersection scenarios
were internally consensus &mdash; meaning every covered demographic cell
(of up to 15) agreed on a Yes-or-No majority among its 5 replicates.
A scenario where 14 cells say Yes and 1 says No is still
<strong>non-consensus</strong>.
</p>
{section1_legend}
{bar1}
{bar2}
{section1_summary}

<h2>2. Cross-model agreement on consensus scenarios</h2>
<p class="section-note">
Taking each model's consensus scenarios as the baseline, how often does
the other model's overall verdict (its majority over its covered cells)
match? "Split" means the other model's covered cells came out tied, so
there's no overall verdict to call agreement or disagreement on.
</p>
{section2_legend}
{bar3}
{bar4}
{section2_summary}

</body></html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--claude", action="append", default=[], metavar="CSV",
                   help="Claude results CSV. Repeat to merge multiple files.")
    p.add_argument("--openai", action="append", default=[], metavar="CSV",
                   help="OpenAI results CSV. Repeat to merge multiple files.")
    p.add_argument("--output", default=None,
                   help="Output HTML path. Default: "
                        "reports/consensus_summary_<claude>_vs_<openai>.html.")
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
    both = claude_set & openai_set

    if not both:
        print("error: no overlapping scenarios between Claude and OpenAI CSVs",
              file=sys.stderr)
        sys.exit(2)

    # Per-model internal consensus over the intersection
    c_consensus_sids = {s for s in both if is_unanimous_partial(claude_rates, s)}
    o_consensus_sids = {s for s in both if is_unanimous_partial(openai_rates, s)}
    c_non = len(both) - len(c_consensus_sids)
    o_non = len(both) - len(o_consensus_sids)

    # Yes / No breakdown of each model's consensus pool
    def yn_breakdown(sids, rates):
        y = sum(1 for s in sids if model_verdict(rates, s) == "Yes")
        n = sum(1 for s in sids if model_verdict(rates, s) == "No")
        return y, n
    c_yn = yn_breakdown(c_consensus_sids, claude_rates)
    o_yn = yn_breakdown(o_consensus_sids, openai_rates)

    # Cross-model: for each model's consensus pool, what does the OTHER say?
    cc_agree, cc_disagree, cc_no_verdict = cross_compare(
        c_consensus_sids, claude_rates, openai_rates)
    oo_agree, oo_disagree, oo_no_verdict = cross_compare(
        o_consensus_sids, openai_rates, claude_rates)

    # Pull model labels from the first row of each side
    def first_model(paths):
        for path in paths:
            with path.open(newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("model"):
                        return r["model"]
        return ""
    claude_model = first_model(claude_paths)
    openai_model = first_model(openai_paths)

    if args.output:
        out = Path(args.output)
    else:
        c_label = _sanitize_for_filename(claude_model or "claude")
        o_label = _sanitize_for_filename(openai_model or "openai")
        out = (Path(__file__).resolve().parent / "reports"
               / f"consensus_summary_{c_label}_vs_{o_label}.html")
    output_path = resolve_non_clobbering(out)
    if output_path != out:
        print(f"warning: {out} exists; writing to {output_path}",
              file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        build_html(
            claude_paths, openai_paths, claude_model, openai_model,
            len(both), len(claude_set - openai_set), len(openai_set - claude_set),
            len(c_consensus_sids), c_non, c_yn,
            len(o_consensus_sids), o_non, o_yn,
            cc_agree, cc_disagree, cc_no_verdict,
            oo_agree, oo_disagree, oo_no_verdict,
        ),
        encoding="utf-8",
    )

    # Stdout summary so the user sees the numbers without opening the HTML
    print(f"intersection: {len(both)} scenarios "
          f"(claude-only {len(claude_set - openai_set)}, "
          f"openai-only {len(openai_set - claude_set)})", file=sys.stderr)
    print(f"  Claude consensus: {len(c_consensus_sids)} / {len(both)} "
          f"(Yes={c_yn[0]}, No={c_yn[1]})", file=sys.stderr)
    print(f"  OpenAI consensus: {len(o_consensus_sids)} / {len(both)} "
          f"(Yes={o_yn[0]}, No={o_yn[1]})", file=sys.stderr)
    print(f"  Claude consensus → OpenAI: agree={cc_agree}, "
          f"disagree={cc_disagree}, split={cc_no_verdict}", file=sys.stderr)
    print(f"  OpenAI consensus → Claude: agree={oo_agree}, "
          f"disagree={oo_disagree}, split={oo_no_verdict}", file=sys.stderr)
    print(f"wrote {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
