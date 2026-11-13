"""
scripts/curate.py

BingeBuddy - build a curated catalog from the processed dataset.
================================================================
Reads data/processed/shows.csv in small chunks and writes
data/processed/shows_curated.csv containing only shows that look good in a demo:

  * a non-blank name
  * vote_count >= MIN_VOTES   (default 10)
  * vote_average > 0
  * non-empty recommendation_text (so the recommender has something to compare)

The input file is opened read-only and is never modified. Values are copied
through as the original text, so nothing is re-formatted or invented.
The output is written to a temporary file first and moved into place only when
the whole run succeeds.

Run:
    .venv/bin/python scripts/curate.py
    .venv/bin/python scripts/curate.py --min-votes 25
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / "data" / "processed" / "shows.csv"
DEFAULT_OUTPUT = BASE_DIR / "data" / "processed" / "shows_curated.csv"
DEFAULT_MIN_VOTES = 10
DEFAULT_CHUNKSIZE = 20_000

REQUIRED = ["id", "name", "vote_average", "vote_count", "recommendation_text"]


def curate_chunk(chunk: pd.DataFrame, min_votes: int) -> tuple[pd.DataFrame, dict]:
    """Filter one chunk (all columns as text). Returns (kept_rows, removal_counts).

    Each removed row is counted once, under the first rule it fails, in this order:
    blank name, too few votes, no rating, no recommendation text.
    """
    votes = pd.to_numeric(chunk["vote_count"], errors="coerce").fillna(0)
    rating = pd.to_numeric(chunk["vote_average"], errors="coerce").fillna(0)

    blank_name = chunk["name"].str.strip() == ""
    few_votes = ~blank_name & (votes < min_votes)
    no_rating = ~blank_name & ~few_votes & (rating <= 0)
    no_text = ~blank_name & ~few_votes & ~no_rating & (chunk["recommendation_text"].str.strip() == "")

    keep = ~(blank_name | few_votes | no_rating | no_text)
    counts = {
        "blank name": int(blank_name.sum()),
        f"fewer than {min_votes} votes": int(few_votes.sum()),
        "no rating": int(no_rating.sum()),
        "no overview/genre text": int(no_text.sum()),
    }
    return chunk[keep], counts


def curate(input_path: Path, output_path: Path, min_votes: int, chunksize: int) -> dict:
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if input_path == output_path:
        raise ValueError("Output path must differ from the input path; the input is never overwritten.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    header = pd.read_csv(input_path, nrows=0).columns.tolist()
    missing = [c for c in REQUIRED if c not in header]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    total_in = 0
    total_out = 0
    removed: dict[str, int] = {}
    first = True

    try:
        reader = pd.read_csv(
            input_path, dtype=str, keep_default_na=False, chunksize=chunksize,
        )
        with open(tmp_path, "w", encoding="utf-8", newline="") as out:
            for chunk in reader:
                total_in += len(chunk)
                kept, counts = curate_chunk(chunk, min_votes)
                for key, val in counts.items():
                    removed[key] = removed.get(key, 0) + val
                kept.to_csv(out, index=False, header=first)
                first = False
                total_out += len(kept)
        if first:  # empty input: still write a header-only file
            pd.DataFrame(columns=header).to_csv(tmp_path, index=False)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {
        "input": input_path,
        "output": output_path,
        "min_votes": min_votes,
        "rows_in": total_in,
        "rows_out": total_out,
        "removed": removed,
        "output_bytes": output_path.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Create data/processed/shows_curated.csv from shows.csv")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--min-votes", type=int, default=DEFAULT_MIN_VOTES)
    ap.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    args = ap.parse_args(argv)

    try:
        r = curate(args.input, args.output, args.min_votes, args.chunksize)
    except (FileNotFoundError, ValueError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print("Curation report")
    print("---------------")
    print(f"Input (unchanged):        {r['input']}")
    print(f"Output:                   {r['output']}")
    print(f"Minimum votes:            {r['min_votes']}")
    print(f"Rows read:                {r['rows_in']:,}")
    print(f"Rows kept:                {r['rows_out']:,}")
    print(f"Rows removed:             {r['rows_in'] - r['rows_out']:,}")
    for reason, n in r["removed"].items():
        print(f"  - {reason}: {n:,}")
    print(f"Output size:              {r['output_bytes'] / 1_048_576:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
