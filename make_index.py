#!/usr/bin/env python3
"""Generate reports/index.html — a master landing page that links every
report in reports/ grouped by report type.

Naming convention drives grouping: a file's prefix determines its section.
Anything that doesn't match a known prefix is bucketed into "Other".

Run after generating any new report:
    python make_index.py
"""

import html as _html
import re
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

# (prefix, section title, one-line description, collapse_default)
# Sections with many files (scenario pairs / single scenarios) collapse by
# default to keep the page scannable.
GROUPS = [
    ("consensus_summary_",
     "Consensus summaries",
     "Per-model internal consensus vs non-consensus and cross-model "
     "agreement on consensus scenarios.",
     False),
    ("cross_model_",
     "Cross-model galleries",
     "Side-by-side 5×3 heatmaps for scenarios where either model is "
     "internally contested or the two models disagree on verdict, with "
     "paired per-cell explanations.",
     False),
    ("contested_gallery_",
     "Contested-scenario galleries (single model)",
     "5×3 heatmap per contested scenario for one model, with "
     "per-cell explanation tables.",
     False),
    ("variant_trends_",
     "Variant trend (similarity) reports",
     "Pairwise-agreement and Spearman-correlation matrices across the "
     "15 race × income variants. Side-by-side panels when two models "
     "are compared. Toggle between plots 1 and 2 in the header.",
     False),
    ("misalignment_",
     "Misalignment-from-consensus reports",
     "Per-group leave-one-out distance from the consensus yes-rate, "
     "ranked by signed bias.",
     False),
    ("scenario_pair_",
     "Scenario-pair comparisons",
     "Two related scenarios viewed side-by-side.",
     True),
    ("scenario_report_",
     "Single-scenario reports",
     "Detail report for a single scenario.",
     True),
]


def esc(v) -> str:
    return _html.escape(str(v) if v is not None else "")


def humanize(name: str, prefix: str) -> str:
    """Strip the group prefix and the .html suffix; lightly format the rest."""
    stem = name[len(prefix):].removesuffix(".html")
    stem = re.sub(r"_(\d{5})-(\d{5})", r" · scenarios \1–\2", stem)
    stem = re.sub(r"_00001_00136-00500", " · scenarios 00001+00136–00500", stem)
    stem = re.sub(r"_vs_", " vs ", stem)
    stem = re.sub(r"_explain-yes\b", " · explain-yes", stem)
    stem = re.sub(r"_explain-no\b", " · explain-no", stem)
    stem = re.sub(r"_axis-(race|income)\b", r" · axis=\1", stem)
    stem = re.sub(r"_recovered\b", " (recovered)", stem)
    stem = stem.replace("_", " ")
    return stem.strip(" ·") or name


CSS = """
:root {
  --bg: #fafafa;
  --card: #ffffff;
  --ink: #1a1a1a;
  --ink-soft: #444;
  --ink-mute: #707070;
  --rule: #e3e3e3;
  --accent: #1a4a8a;
  --accent-soft: #eaf0fb;
  --mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  max-width: 1080px; margin: 0 auto; padding: 36px 24px 80px;
  color: var(--ink); background: var(--bg);
  line-height: 1.55;
}
header.page {
  border-bottom: 1px solid var(--rule);
  padding-bottom: 18px;
  margin-bottom: 26px;
}
header.page h1 {
  font-size: 2em; margin: 0 0 6px 0; letter-spacing: -0.01em;
}
header.page .subtitle {
  color: var(--ink-mute); font-size: 1.02em;
  max-width: 64ch;
}
header.page code { background: #f0f0f0; padding: 1px 6px;
                    border-radius: 4px; font-family: var(--mono);
                    font-size: 0.9em; }

.stats {
  display: flex; flex-wrap: wrap; gap: 22px;
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 8px; padding: 14px 18px; margin: 0 0 28px 0;
}
.stats .stat-num {
  font-size: 1.5em; font-weight: 700; color: var(--accent);
  font-variant-numeric: tabular-nums; line-height: 1.1;
}
.stats .stat-lbl {
  font-size: 0.82em; color: var(--ink-mute);
  text-transform: uppercase; letter-spacing: 0.04em;
  margin-top: 2px;
}

section.group {
  background: var(--card); border: 1px solid var(--rule);
  border-radius: 8px; margin: 0 0 18px 0; overflow: hidden;
}
section.group > summary, section.group > .group-head {
  display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
  padding: 14px 20px; cursor: default;
  background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 100%);
  border-bottom: 1px solid var(--rule);
}
details.group > summary {
  cursor: pointer; list-style: none;
}
details.group > summary::-webkit-details-marker { display: none; }
details.group > summary::before {
  content: "▸"; color: var(--ink-mute); display: inline-block;
  width: 14px; transition: transform 0.15s ease;
  font-size: 0.95em;
}
details.group[open] > summary::before { transform: rotate(90deg); }

section.group h2, details.group summary h2 {
  font-size: 1.1em; margin: 0; color: var(--ink);
  font-weight: 600;
}
.count-pill {
  display: inline-block; padding: 2px 9px; border-radius: 999px;
  background: var(--accent-soft); color: var(--accent);
  font-size: 0.8em; font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.group-desc {
  width: 100%; color: var(--ink-mute); font-size: 0.9em;
  margin-top: 2px; line-height: 1.5;
}
ul.report-list {
  list-style: none; padding: 8px 8px 12px 8px; margin: 0;
}
ul.report-list li {
  padding: 7px 12px; border-radius: 6px;
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px;
  transition: background 0.1s ease;
}
ul.report-list li:hover { background: var(--accent-soft); }
ul.report-list a {
  color: var(--accent); text-decoration: none; font-weight: 500;
  font-size: 0.98em;
}
ul.report-list a:hover { text-decoration: underline; }
ul.report-list .fname {
  color: var(--ink-mute); font-size: 0.78em;
  font-family: var(--mono);
  margin-left: auto; padding-left: 12px;
}
footer.page {
  margin-top: 36px; padding-top: 16px;
  border-top: 1px solid var(--rule);
  color: var(--ink-mute); font-size: 0.85em;
}
footer.page code { font-family: var(--mono); background: #f0f0f0;
                    padding: 1px 5px; border-radius: 3px; }

@media (max-width: 640px) {
  body { padding: 20px 14px 60px; }
  header.page h1 { font-size: 1.55em; }
  ul.report-list .fname { display: none; }
  ul.report-list li { padding: 8px 10px; }
}
"""


def render_section(title: str, desc: str, prefix: str,
                   names: list[str], collapsed: bool) -> str:
    tag = "details" if collapsed else "section"
    open_attr = "" if collapsed else ""
    items = []
    for name in names:
        label = humanize(name, prefix) if prefix else name
        items.append(
            f'<li><a href="{esc(name)}">{esc(label)}</a>'
            f'<span class="fname">{esc(name)}</span></li>'
        )
    head_block = (
        f'<h2>{esc(title)}</h2>'
        f'<span class="count-pill">{len(names)}</span>'
        f'<div class="group-desc">{esc(desc)}</div>'
    )
    if collapsed:
        return (
            f'<details class="group" {open_attr}>'
            f'<summary>{head_block}</summary>'
            f'<ul class="report-list">{"".join(items)}</ul>'
            f'</details>'
        )
    else:
        return (
            f'<section class="group">'
            f'<div class="group-head">{head_block}</div>'
            f'<ul class="report-list">{"".join(items)}</ul>'
            f'</section>'
        )


def main() -> None:
    files = sorted(p.name for p in REPORTS_DIR.glob("*.html")
                   if p.name != "index.html")

    sections: list[tuple[str, str, str, list[str], bool]] = []
    assigned: set[str] = set()
    for prefix, title, desc, collapsed in GROUPS:
        matches = [f for f in files if f.startswith(prefix)]
        if matches:
            sections.append((title, desc, prefix, matches, collapsed))
            assigned.update(matches)

    other = [f for f in files if f not in assigned]
    if other:
        sections.append(("Other",
                         "Reports that don't match a known prefix.",
                         "", other, False))

    total = sum(len(names) for _, _, _, names, _ in sections)

    parts: list[str] = [f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>510 final project — report index</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{CSS}</style></head><body>
<header class="page">
<h1>510 final project — report index</h1>
<div class="subtitle">
Generated reports for the moral-scenario demographic-bias study. Reports
live under <code>reports/</code> and are grouped here by type. Long
sections (scenario pairs, single scenarios) are collapsed by default.
</div>
</header>
<div class="stats">
<div><div class="stat-num">{total}</div><div class="stat-lbl">total reports</div></div>
<div><div class="stat-num">{len(sections)}</div><div class="stat-lbl">sections</div></div>
</div>
"""]

    for title, desc, prefix, names, collapsed in sections:
        parts.append(render_section(title, desc, prefix, names, collapsed))

    parts.append(
        '<footer class="page">Regenerate with '
        '<code>python make_index.py</code> after producing any new report.'
        '</footer></body></html>'
    )

    out = REPORTS_DIR / "index.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out} ({total} reports across {len(sections)} sections)")


if __name__ == "__main__":
    main()
