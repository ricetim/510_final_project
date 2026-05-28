#!/usr/bin/env python3
"""Recover Yes/No answers from a run_claude_text.py results CSV.

When the text-mode runner gets prose like "Yes.\\n\\nReasoning..." back from
the model (common with --thinking on), its strict parser rejects the whole
blob and writes `error="unexpected output: <prose>"` with `answer=""`. This
script scans those rows, takes the first alphabetic word of the prose
(skipping leading whitespace, markdown markers like `#` `*`, and punctuation),
and if it's "yes" or "no":

  - promotes it to the `answer` column,
  - moves the remaining prose (post-first-word, with leading punctuation and
    whitespace stripped) into the `explanation` column,
  - clears the `error` column so the row passes downstream "skip if error"
    gates (e.g. extract_high_spread.py).

Rows whose first word is something else (e.g. "I can't answer...") are left
untouched. The full original prose was stored upstream by run_claude_text.py;
older CSVs produced before that fix were truncated to 100 chars, so the
salvaged `explanation` on those rows will be visibly cut off — that's a
property of the source data, not this script.

Usage:
    python recover_answers.py results/<file>.csv [--output <new_path>]

Default output: inserts `_recovered` before the extension, next to the input.
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

ERROR_PREFIX = "unexpected output: "
FIRST_WORD_RE = re.compile(r"^[\s#*_>`\-:.,!?'\"()\[\]]*([A-Za-z]+)")
# After we lop off the first yes/no word, trim leading punctuation and
# whitespace from what's left so the explanation reads naturally.
LEADING_JUNK_RE = re.compile(r"^[\s.,;:!?)\]]+")


def split_first_word(prose: str) -> tuple[str, str]:
    """Return (first_word_lower, cleaned_remainder). Empty strings if no match."""
    m = FIRST_WORD_RE.match(prose)
    if not m:
        return "", ""
    remainder = LEADING_JUNK_RE.sub("", prose[m.end():])
    return m.group(1).lower(), remainder


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("results_csv", help="Path to a results CSV with empty-answer rows.")
    p.add_argument("--output", default=None,
                   help="Output CSV path. Default: <name>_recovered.csv next to the input.")
    args = p.parse_args()

    src = Path(args.results_csv)
    if not src.exists():
        print(f"error: results CSV not found: {src}", file=sys.stderr)
        sys.exit(2)

    out = Path(args.output) if args.output \
        else src.with_name(f"{src.stem}_recovered{src.suffix}")

    with src.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames or "answer" not in fieldnames or "error" not in fieldnames:
        print("error: input CSV missing required columns (answer, error).",
              file=sys.stderr)
        sys.exit(2)

    recovered_yes = recovered_no = 0
    already_filled = no_prefix = no_match = 0
    first_words: Counter[str] = Counter()

    for r in rows:
        if r["answer"].strip():
            already_filled += 1
            continue
        err = r.get("error", "")
        if not err.startswith(ERROR_PREFIX):
            no_prefix += 1
            continue
        prose = err[len(ERROR_PREFIX):]
        w, remainder = split_first_word(prose)
        first_words[w or "(none)"] += 1
        if w == "yes":
            r["answer"] = "Yes"
            r["explanation"] = remainder
            r["error"] = ""
            recovered_yes += 1
        elif w == "no":
            r["answer"] = "No"
            r["explanation"] = remainder
            r["error"] = ""
            recovered_no += 1
        else:
            no_match += 1

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    recovered = recovered_yes + recovered_no
    print(f"source:       {src}", file=sys.stderr)
    print(f"destination:  {out}", file=sys.stderr)
    print(f"total rows:   {total}", file=sys.stderr)
    print(f"already filled (unchanged):           {already_filled}", file=sys.stderr)
    print(f"recovered as Yes:                     {recovered_yes}", file=sys.stderr)
    print(f"recovered as No:                      {recovered_no}", file=sys.stderr)
    print(f"recovered total:                      {recovered}", file=sys.stderr)
    print(f"not recovered (first word not yes/no): {no_match}", file=sys.stderr)
    print(f"not recovered (no 'unexpected output:' prefix): {no_prefix}",
          file=sys.stderr)
    if first_words:
        print("\nfirst-word distribution across empty-answer rows:", file=sys.stderr)
        for w, c in first_words.most_common():
            print(f"  {c:>5}  {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
