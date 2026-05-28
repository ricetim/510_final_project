#!/usr/bin/env python3
"""Find which demographic variants answer together (and which oppose).

Takes a results CSV and emits a single HTML report with four views:

  1. Pairwise agreement matrix — 15x15 heatmap of "fraction of scenarios on
     which these two variants gave the same majority answer." Diagonal = 1.
  2. Spearman correlation matrix — 15x15 heatmap of rank correlation on
     per-scenario yes-rates. Diverging color scale: blue = answer together,
     red = answer opposite. Captures partial agreement that #1 misses.
  3. Hierarchical clustering dendrogram — average linkage on (1 - Spearman)
     distances. Reveals natural demographic blocs at multiple resolutions.
  4. Race-vs-income decomposition — three numbers: mean pair-distance for
     same-race pairs, same-income pairs, and fully-different pairs. Tells
     you whether the bias is mostly race-shaped or income-shaped.

Usage:
    python variant_trends_report.py <results_csv> [--output report.html]
                                                  [--include-unanimous]

Scenarios where all 15 variants give the SAME majority answer (everyone
Yes, or everyone No) carry no demographic signal — they pull every pair's
agreement / correlation toward each other without telling you anything
about bias. By default they're dropped before the matrices are built;
pass --include-unanimous to keep them.

Default output: reports/variant_trends_<model>.html, anchored to the
script's parent dir (not CWD), no-clobber.

No numpy/scipy required. Everything is implemented in pure Python because
n=15 variants is small enough that O(n^2) and O(n^3) routines run in
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


def esc(value) -> str:
    return _html.escape(str(value) if value is not None else "")


def _sanitize_for_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value) or "unknown"


def default_output_path(model: str) -> Path:
    return Path(__file__).resolve().parent / "reports" \
        / f"variant_trends_{_sanitize_for_filename(model)}.html"


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

def load_yes_rate_matrix(path: Path) -> tuple[list[str], dict[tuple[str, str], dict[str, float]]]:
    """Return (scenario_ids_sorted, variant -> {scenario_id: yes_rate}).

    Only scenarios where ALL 15 variants have at least one Yes/No reply are
    included — anything else would create asymmetric NaNs that infect the
    pairwise math downstream. This is conservative; on the recovered haiku
    CSV it keeps ~95% of scenarios.
    """
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    # variant -> scenario -> [answers]
    raw: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    all_scenarios: set[str] = set()
    for r in rows:
        if r["answer"] not in ("Yes", "No"):
            continue
        key = (r["race_variant"], r["income_variant"])
        if key not in set(VARIANTS):
            continue
        raw[key][r["scenario_id"]].append(r["answer"])
        all_scenarios.add(r["scenario_id"])

    # Compute yes-rates
    rates: dict[tuple[str, str], dict[str, float]] = {v: {} for v in VARIANTS}
    for v in VARIANTS:
        for sid, answers in raw[v].items():
            rates[v][sid] = sum(1 for a in answers if a == "Yes") / len(answers)

    # Keep only scenarios where all 15 variants have data.
    complete = sorted(
        sid for sid in all_scenarios
        if all(sid in rates[v] for v in VARIANTS)
    )
    rates = {v: {sid: rates[v][sid] for sid in complete} for v in VARIANTS}
    return complete, rates


# ---------------------------------------------------------------------------
# Method 1: pairwise agreement on majority votes
# ---------------------------------------------------------------------------

def majority(yes_rate: float) -> str | None:
    if yes_rate > 0.5:
        return "Yes"
    if yes_rate < 0.5:
        return "No"
    return None  # 0.5 = ambiguous; exclude from this method


def is_unanimous(rates: dict[tuple[str, str], dict[str, float]], sid: str) -> bool:
    """True if all 15 variants give the same Yes/No majority on this scenario.

    A variant with an exactly-50/50 yes-rate (no majority) prevents the
    scenario from being unanimous — we can't say it agrees with anything.
    """
    seen: set[str] = set()
    for v in VARIANTS:
        m = majority(rates[v][sid])
        if m is None:
            return False
        seen.add(m)
    return len(seen) == 1


def agreement_matrix(scenarios: list[str],
                     rates: dict[tuple[str, str], dict[str, float]]) -> list[list[float]]:
    n = len(VARIANTS)
    out = [[0.0] * n for _ in range(n)]
    for i, vi in enumerate(VARIANTS):
        for j, vj in enumerate(VARIANTS):
            if i == j:
                out[i][j] = 1.0
                continue
            agree = total = 0
            for sid in scenarios:
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
                    rates: dict[tuple[str, str], dict[str, float]]) -> list[list[float]]:
    n = len(VARIANTS)
    vecs = [[rates[v][sid] for sid in scenarios] for v in VARIANTS]
    ranks = [midranks(v) for v in vecs]
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            out[i][j] = 1.0 if i == j else pearson(ranks[i], ranks[j])
    return out


# ---------------------------------------------------------------------------
# Method 3: hierarchical clustering (average linkage) + SVG dendrogram
# ---------------------------------------------------------------------------

def average_linkage(distance: list[list[float]]) -> list[tuple[int, int, float, int]]:
    """Return linkage matrix [(left, right, height, total_size), ...].

    Cluster ids 0..n-1 are leaves; ids n..2n-2 are internal nodes. The
    distance table is mutated in place via Lance-Williams (average linkage).
    """
    n = len(distance)
    # dist[(a,b)] for ordered (a<b in id, but ids are not contiguous so use raw)
    dist: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = distance[i][j]

    sizes = {i: 1 for i in range(n)}
    active = list(range(n))
    linkage: list[tuple[int, int, float, int]] = []
    next_id = n

    def key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    for _ in range(n - 1):
        # Find closest pair among active clusters.
        best = None
        best_d = float("inf")
        for a_idx in range(len(active)):
            a = active[a_idx]
            for b_idx in range(a_idx + 1, len(active)):
                b = active[b_idx]
                d = dist[key(a, b)]
                if d < best_d:
                    best_d = d
                    best = (a, b)
        assert best is not None
        a, b = best
        new_id = next_id
        next_id += 1
        new_size = sizes[a] + sizes[b]
        linkage.append((a, b, best_d, new_size))

        # Lance-Williams for average linkage:
        # d(new, x) = (size_a * d(a,x) + size_b * d(b,x)) / (size_a + size_b)
        for x in active:
            if x == a or x == b:
                continue
            d_ax = dist[key(a, x)]
            d_bx = dist[key(b, x)]
            dist[key(new_id, x)] = (sizes[a] * d_ax + sizes[b] * d_bx) / new_size

        sizes[new_id] = new_size
        active.remove(a)
        active.remove(b)
        active.append(new_id)
    return linkage


def dendrogram_svg(linkage: list[tuple[int, int, float, int]],
                   labels: list[str],
                   width: int = 900, height: int = 480) -> str:
    """Render a horizontal dendrogram as SVG. Leaves on the bottom, root on top."""
    n = len(labels)
    # Build tree: child_of[node] -> (left, right, height)
    children: dict[int, tuple[int, int, float]] = {}
    for i, (a, b, h, _) in enumerate(linkage):
        children[n + i] = (a, b, h)

    # Determine leaf order via in-order DFS of the merge tree.
    leaf_order: list[int] = []

    def dfs(node: int) -> None:
        if node < n:
            leaf_order.append(node)
            return
        a, b, _h = children[node]
        dfs(a)
        dfs(b)

    root = n + len(linkage) - 1
    dfs(root)

    # Layout: x = leaf order position; y = merge height (root at top).
    label_band = 130  # px reserved at the bottom for rotated labels
    plot_h = height - label_band - 20
    plot_w = width - 40
    left_pad = 20

    x_of_leaf = {leaf: left_pad + (i + 0.5) * plot_w / n for i, leaf in enumerate(leaf_order)}
    max_h = max(h for _, _, h, _ in linkage) or 1.0

    def y_of_height(h: float) -> float:
        # h=0 at bottom (just above labels), h=max_h at top.
        return 10 + (1 - h / max_h) * plot_h

    # Compute x for each cluster (leaf or internal) by recursion.
    x_of: dict[int, float] = dict(x_of_leaf)
    for i, (a, b, _h, _) in enumerate(linkage):
        x_of[n + i] = (x_of[a] + x_of[b]) / 2

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'class="dendrogram" font-family="sans-serif" font-size="11">'
    )
    # Lines
    for i, (a, b, h, _) in enumerate(linkage):
        xa, xb = x_of[a], x_of[b]
        ya = y_of_height(children[a][2]) if a in children else y_of_height(0)
        yb = y_of_height(children[b][2]) if b in children else y_of_height(0)
        yh = y_of_height(h)
        # Two vertical bars + one horizontal at merge height.
        parts.append(f'<line x1="{xa:.1f}" y1="{ya:.1f}" x2="{xa:.1f}" y2="{yh:.1f}" stroke="#333" stroke-width="1.2"/>')
        parts.append(f'<line x1="{xb:.1f}" y1="{yb:.1f}" x2="{xb:.1f}" y2="{yh:.1f}" stroke="#333" stroke-width="1.2"/>')
        parts.append(f'<line x1="{xa:.1f}" y1="{yh:.1f}" x2="{xb:.1f}" y2="{yh:.1f}" stroke="#333" stroke-width="1.2"/>')

    # Leaf labels (rotated -45°).
    y_baseline = y_of_height(0) + 8
    for leaf in leaf_order:
        x = x_of_leaf[leaf]
        parts.append(
            f'<text x="{x:.1f}" y="{y_baseline:.1f}" text-anchor="end" '
            f'transform="rotate(-45 {x:.1f} {y_baseline:.1f})">{esc(labels[leaf])}</text>'
        )

    # Y-axis ticks (distance = 1 - rho range).
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        if tick > max_h * 1.05:
            continue
        y = y_of_height(tick)
        parts.append(f'<line x1="{left_pad - 5}" y1="{y:.1f}" x2="{left_pad}" y2="{y:.1f}" stroke="#666"/>')
        parts.append(f'<text x="{left_pad - 8}" y="{y + 3:.1f}" text-anchor="end" fill="#666" font-size="10">{tick:.2f}</text>')

    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Method 4: race-vs-income decomposition
# ---------------------------------------------------------------------------

def decomposition(distance: list[list[float]]) -> dict[str, float]:
    """Average pairwise distance grouped by what's shared between the pair."""
    same_race: list[float] = []
    same_income: list[float] = []
    different: list[float] = []
    for i, vi in enumerate(VARIANTS):
        for j, vj in enumerate(VARIANTS):
            if i >= j:
                continue
            d = distance[i][j]
            if vi[0] == vj[0]:
                same_race.append(d)
            elif vi[1] == vj[1]:
                same_income.append(d)
            else:
                different.append(d)

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "same_race": avg(same_race),
        "same_income": avg(same_income),
        "different": avg(different),
        "n_same_race": len(same_race),
        "n_same_income": len(same_income),
        "n_different": len(different),
    }


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


def render_matrix(matrix: list[list[float]],
                  labels: list[str],
                  color_fn,
                  fmt: str = "{:.2f}") -> str:
    parts: list[str] = ['<table class="matrix"><thead><tr><th></th>']
    for lbl in labels:
        parts.append(f'<th class="rot"><div><span>{esc(lbl)}</span></div></th>')
    parts.append("</tr></thead><tbody>")
    for i, lbl in enumerate(labels):
        parts.append(f'<tr><th class="row">{esc(lbl)}</th>')
        for j in range(len(labels)):
            v = matrix[i][j]
            cell = "—" if math.isnan(v) else fmt.format(v)
            parts.append(
                f'<td style="background:{color_fn(v)};">{esc(cell)}</td>'
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
.note { color: #555; font-size: 0.9em; max-width: 78ch; margin: 0.5em 0 1em 0; }
table.matrix { border-collapse: collapse; margin: 0.5em 0; font-size: 0.78em; }
table.matrix th, table.matrix td { border: 1px solid #ccc; padding: 4px 6px;
                                   text-align: center; font-variant-numeric: tabular-nums; }
table.matrix th.row { text-align: right; font-weight: 500; background: #f3f3f3; }
table.matrix th.rot { vertical-align: bottom; height: 110px; padding: 0; }
table.matrix th.rot > div { transform: rotate(-55deg); transform-origin: bottom left;
                            width: 20px; margin-left: 6px; white-space: nowrap; }
table.matrix th.rot > div > span { padding: 2px 4px; }
.dendrogram { background: #fafafa; border: 1px solid #ddd; max-width: 100%; }
.deco { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1em;
        margin: 1em 0 0.5em 0; }
.deco .cell { background: #f7f7f7; padding: 0.8em 1em; border-left: 4px solid #888;
              text-align: center; }
.deco .cell .num { font-size: 1.6em; font-variant-numeric: tabular-nums;
                   color: #1a4a8a; }
.deco .cell .lbl { font-size: 0.85em; color: #555; margin-top: 0.3em; }
.legend { display: inline-flex; gap: 1.2em; align-items: center; font-size: 0.85em;
          color: #555; margin: 0.5em 0; }
.legend .swatch { display: inline-block; width: 18px; height: 12px; vertical-align: middle;
                  border: 1px solid #aaa; margin-right: 4px; }
"""


def build_html(scenarios: list[str], rates, agree, corr, linkage, deco,
               results_path: Path, model: str,
               n_complete: int, unanimous_dropped: int) -> str:
    parts: list[str] = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Variant trends</title>
<style>{CSS}</style></head><body>""")
    parts.append("<h1>Variant trend analysis</h1>")
    filter_note = (
        f"{unanimous_dropped} unanimous-consensus scenarios dropped"
        if unanimous_dropped else "unanimous-consensus scenarios included"
    )
    parts.append(
        f'<div class="meta">Source: <code>{esc(results_path)}</code> · '
        f'Model: <code>{esc(model)}</code> · '
        f'{len(scenarios)} of {n_complete} full-coverage scenarios used '
        f'({esc(filter_note)})</div>'
    )

    # Method 1
    parts.append("<h2>1. Pairwise agreement matrix</h2>")
    parts.append(
        '<div class="note">Each cell: fraction of scenarios on which these two '
        "variants gave the same majority answer (Yes-rate &gt; 0.5 vs &lt; 0.5). "
        "Diagonal is trivially 1.00. Read hot cells as &quot;answer together,&quot; "
        "cold cells as &quot;answer opposite.&quot;</div>"
    )
    parts.append(
        '<div class="legend">'
        '<span><span class="swatch" style="background:hsl(220,60%,100%)"></span>0.0 (always disagree)</span>'
        '<span><span class="swatch" style="background:hsl(220,60%,72%)"></span>0.5</span>'
        '<span><span class="swatch" style="background:hsl(220,60%,45%)"></span>1.0 (always agree)</span>'
        '</div>'
    )
    parts.append(render_matrix(agree, VARIANT_LABELS, hsl_for_agreement))

    # Method 2
    parts.append("<h2>2. Spearman rank correlation</h2>")
    parts.append(
        '<div class="note">Each cell: Spearman &rho; on per-scenario yes-rates. '
        "Captures partial agreement that the majority-vote matrix above misses "
        "(e.g. variant A at 0.8/0.6/0.2 vs B at 0.7/0.5/0.1 → high &rho; even "
        "though all majorities are tied). Range &minus;1 (perfectly opposite) "
        "to +1 (perfectly aligned).</div>"
    )
    parts.append(
        '<div class="legend">'
        '<span><span class="swatch" style="background:hsl(0,60%,45%)"></span>&minus;1</span>'
        '<span><span class="swatch" style="background:hsl(0,60%,100%)"></span>0</span>'
        '<span><span class="swatch" style="background:hsl(220,60%,45%)"></span>+1</span>'
        '</div>'
    )
    parts.append(render_matrix(corr, VARIANT_LABELS, hsl_for_corr))

    # Method 3
    parts.append("<h2>3. Hierarchical clustering dendrogram</h2>")
    parts.append(
        '<div class="note">Average linkage on (1 &minus; Spearman &rho;) distances. '
        "Variants that merge low on the y-axis are most similar in answer pattern. "
        "Cut the tree at any height to read off &quot;at this similarity threshold, "
        "the model treats these variants as one bloc.&quot;</div>"
    )
    # Build distance matrix (1 - corr).
    distance = [[1 - corr[i][j] for j in range(len(VARIANTS))]
                for i in range(len(VARIANTS))]
    parts.append(dendrogram_svg(linkage, VARIANT_LABELS, width=1100, height=520))

    # Method 4
    parts.append("<h2>4. Race-vs-income decomposition</h2>")
    parts.append(
        '<div class="note">Mean (1 &minus; Spearman &rho;) distance for pairs of '
        "variants that share <strong>race</strong> (n = "
        f'{deco["n_same_race"]}), share <strong>income</strong> '
        f'(n = {deco["n_same_income"]}), or share <strong>neither</strong> '
        f'(n = {deco["n_different"]}). The lower number is the tighter cluster. '
        "If same-race pairs are tighter than same-income pairs, race is the "
        "dominant grouping axis (and vice versa).</div>"
    )
    parts.append('<div class="deco">')
    for key, label in (
        ("same_race", "Same race, different income"),
        ("same_income", "Same income, different race"),
        ("different", "Different race AND income"),
    ):
        parts.append(
            f'<div class="cell"><div class="num">{deco[key]:.3f}</div>'
            f'<div class="lbl">{esc(label)}</div></div>'
        )
    parts.append("</div>")

    # One-line verdict.
    sr, si = deco["same_race"], deco["same_income"]
    if not math.isnan(sr) and not math.isnan(si) and sr != si:
        if sr < si:
            ratio = si / sr if sr > 0 else float("inf")
            verdict = (f"<strong>Race dominates:</strong> same-race pairs are "
                       f"{ratio:.2f}× tighter than same-income pairs.")
        else:
            ratio = sr / si if si > 0 else float("inf")
            verdict = (f"<strong>Income dominates:</strong> same-income pairs are "
                       f"{ratio:.2f}× tighter than same-race pairs.")
        parts.append(f'<div class="note">{verdict}</div>')

    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("results_csv", help="Path to results CSV.")
    p.add_argument("--output", default=None,
                   help="Output HTML path. Default: reports/variant_trends_<model>.html.")
    p.add_argument("--include-unanimous", action="store_true",
                   help="Keep scenarios where all 15 variants give the same "
                        "majority answer (dropped by default — no demographic "
                        "signal but they pull every pairwise correlation toward "
                        "each other).")
    args = p.parse_args()

    results_path = Path(args.results_csv)
    if not results_path.exists():
        print(f"error: results CSV not found: {results_path}", file=sys.stderr)
        sys.exit(2)

    scenarios, rates = load_yes_rate_matrix(results_path)
    n_complete = len(scenarios)

    unanimous_count = 0
    if not args.include_unanimous:
        unanimous_set = {sid for sid in scenarios if is_unanimous(rates, sid)}
        unanimous_count = len(unanimous_set)
        scenarios = [sid for sid in scenarios if sid not in unanimous_set]
        rates = {v: {sid: rates[v][sid] for sid in scenarios} for v in VARIANTS}
        if unanimous_count:
            print(
                f"dropped {unanimous_count} unanimous-consensus scenarios "
                f"({unanimous_count / n_complete:.0%} of {n_complete}); "
                f"{len(scenarios)} remain. Use --include-unanimous to keep them.",
                file=sys.stderr,
            )

    if len(scenarios) < 5:
        print(
            f"error: only {len(scenarios)} scenarios remain after filtering — "
            "not enough signal for trend analysis. Need at least 5. "
            "Try --include-unanimous if most scenarios were dropped.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Read model name from any row for the title.
    with results_path.open(newline="") as f:
        model = next(
            (r.get("model", "") for r in csv.DictReader(f) if r.get("model")), ""
        )

    agree = agreement_matrix(scenarios, rates)
    corr = spearman_matrix(scenarios, rates)
    distance = [[1 - corr[i][j] for j in range(len(VARIANTS))]
                for i in range(len(VARIANTS))]
    linkage = average_linkage(distance)
    deco = decomposition(distance)

    requested = Path(args.output) if args.output else default_output_path(model)
    output_path = resolve_non_clobbering(requested)
    if output_path != requested:
        print(f"warning: {requested} exists; writing to {output_path} instead",
              file=sys.stderr)

    html = build_html(scenarios, rates, agree, corr, linkage, deco,
                      results_path, model, n_complete, unanimous_count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"wrote 4-method trend report ({len(scenarios)} scenarios) to {output_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
