"""Tiny SVG previews for the report index page.

Strategy: parse each report's HTML for `<code>...csv</code>` source-path
references, group those CSVs by model name, and render one 5×3 mini
heatmap per model — colored by mean yes-rate per (race, income) cell
across that model's data. So the preview answers "what data is this
report drawn from?" rather than "what does this specific analysis
look like" — but the data signature is distinct between Claude vs
OpenAI runs (cells lean differently), so reports source-discriminate
visually at a glance.

Reports whose CSV references can't be resolved (e.g. file missing,
or no CSV reference at all) return None and get no preview.

Caching: CSV loads (the expensive part) are LRU-cached by absolute
path string, so the same CSV referenced by multiple reports only
hits disk once across all preview calls in one index build.
"""

import csv
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from contested_gallery_report import yr_color
from variant_trends_report import RACES, INCOMES, VARIANTS

PROJECT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Cached CSV loaders
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _raw_counts(path_str: str) -> dict:
    """{(race, income): [n_yes, n_no]} for one CSV. Cached on string key."""
    raw: dict = defaultdict(lambda: [0, 0])
    with open(path_str, newline="") as f:
        for r in csv.DictReader(f):
            if r["answer"] not in ("Yes", "No"):
                continue
            if r["race_variant"] not in RACES or r["income_variant"] not in INCOMES:
                continue
            i = 0 if r["answer"] == "Yes" else 1
            raw[(r["race_variant"], r["income_variant"])][i] += 1
    return dict(raw)


@lru_cache(maxsize=32)
def _model_name(path_str: str) -> str:
    """First non-empty model column value in the CSV."""
    with open(path_str, newline="") as f:
        for r in csv.DictReader(f):
            m = r.get("model", "")
            if m:
                return m
    return ""


def _pooled_yes_rates(paths: list[Path]) -> dict:
    """Pool yes/no counts from N CSVs, then convert to yes-rate per cell."""
    raw: dict = defaultdict(lambda: [0, 0])
    for p in paths:
        for k, (y, n) in _raw_counts(str(p)).items():
            raw[k][0] += y
            raw[k][1] += n
    out = {}
    for v in VARIANTS:
        y, n = raw[v]
        out[v] = y / (y + n) if (y + n) else float("nan")
    return out


# ---------------------------------------------------------------------------
# HTML scraping
# ---------------------------------------------------------------------------

_CSV_REF_RE = re.compile(r"<code>([^<]+\.csv)</code>")


def _extract_csv_paths(html_path: Path) -> list[Path]:
    """All distinct .csv references inside <code>…</code>, in order of
    appearance. Only the first 64 KB of the report is scanned — CSV refs
    live in the meta block at the top, and some gallery HTMLs are 12+ MB.
    Non-existent paths are dropped."""
    with html_path.open("rb") as f:
        head = f.read(65536).decode("utf-8", errors="replace")
    seen: set[str] = set()
    out: list[Path] = []
    for raw in _CSV_REF_RE.findall(head):
        s = raw.strip()
        if s in seen:
            continue
        seen.add(s)
        # Resolve against a few candidate locations: absolute, project-root,
        # and project_root/results/ (older reports stored only the basename).
        candidates = (
            [Path(s)] if Path(s).is_absolute() else
            [PROJECT_DIR / s, PROJECT_DIR / "results" / s]
        )
        for p in candidates:
            if p.exists():
                out.append(p)
                break
    return out


def _group_by_model(paths: list[Path]) -> list[tuple[str, list[Path]]]:
    """Group CSVs by model name, preserving the order each model first
    appears. Returns [(model_name, [paths]), …]."""
    order: list[str] = []
    by_model: dict[str, list[Path]] = {}
    for p in paths:
        m = _model_name(str(p)) or "?"
        if m not in by_model:
            order.append(m)
            by_model[m] = []
        by_model[m].append(p)
    return [(m, by_model[m]) for m in order]


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------

def _mini_grid_svg(yes_rates: dict, cell_w: int = 11, cell_h: int = 9) -> str:
    """5×3 mini heatmap (races × incomes), one rect per cell."""
    parts: list[str] = []
    for ii, income in enumerate(INCOMES):
        for ri, race in enumerate(RACES):
            yr = yes_rates.get((race, income), float("nan"))
            if yr != yr:  # NaN
                fill = "#ececec"
            else:
                fill = yr_color(yr)
            parts.append(
                f'<rect x="{ri * cell_w}" y="{ii * cell_h}" '
                f'width="{cell_w}" height="{cell_h}" fill="{fill}"/>'
            )
    w, h = 5 * cell_w, 3 * cell_h
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'preserveAspectRatio="xMidYMid meet" '
        f'shape-rendering="crispEdges">'
        + "".join(parts) +
        '</svg>'
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def preview_for(html_path: Path) -> str | None:
    """Return an inline HTML/SVG snippet that serves as a thumbnail preview
    for the given report. Returns None if the report has no resolvable CSV
    references (so the index can fall back to no-preview gracefully)."""
    paths = _extract_csv_paths(html_path)
    if not paths:
        return None
    grouped = _group_by_model(paths)
    # Cap at 2 panels for layout; pool multiple files per model.
    grouped = grouped[:2]
    svgs = [_mini_grid_svg(_pooled_yes_rates(model_paths))
            for _, model_paths in grouped]
    if len(svgs) == 1:
        return f'<span class="preview-wrap">{svgs[0]}</span>'
    return (
        '<span class="preview-wrap preview-pair">'
        + "".join(f'<span>{s}</span>' for s in svgs) +
        '</span>'
    )
