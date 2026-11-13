"""
src/data_loader.py

BingeBuddy - Phase 3 Data Loader
=================================
Loads the preprocessed TMDB TV shows dataset into memory safely and efficiently.

Design decisions for 4 GB RAM:
- Only the 12 confirmed columns from data/processed/shows.csv are loaded.
- Numeric columns are downcast to narrow types (float32, int32, int16) at read time.
- A single DataFrame is stored; no whole-dataset copies are created.
- An id→positional-index lookup dict enables O(1) row access without scanning.
- The module validates uniqueness and required columns on first load, not on every call.
- Column names match exactly what preprocess.py wrote; do not assume extra fields.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
FULL_CSV = BASE_DIR / "data" / "processed" / "shows.csv"
CURATED_CSV = BASE_DIR / "data" / "processed" / "shows_curated.csv"

# Use the curated catalog (made by scripts/curate.py) when it exists; otherwise fall
# back to the full processed file. Set BINGEBUDDY_DATASET=full to force the full file.
if CURATED_CSV.exists() and os.environ.get("BINGEBUDDY_DATASET", "").lower() != "full":
    PROCESSED_CSV = CURATED_CSV
else:
    PROCESSED_CSV = FULL_CSV

# ---------------------------------------------------------------------------
# Column configuration
# ---------------------------------------------------------------------------
# Columns actually present in data/processed/shows.csv (confirmed by inspection)
REQUIRED_COLUMNS: list[str] = [
    "id",
    "name",
    "overview",
    "genres",
    "vote_average",
    "vote_count",
    "popularity",
    "first_air_date",
    "poster_path",
    "number_of_seasons",
    "number_of_episodes",
    "recommendation_text",
]

# Memory-efficient dtypes. Strings stay as object (pandas default); narrowing
# numerics saves ~30-40 % RAM vs. int64/float64 for 165 k rows.
_DTYPE_MAP: dict[str, str] = {
    "id":                 "int64",   # must stay int64 for safe dict keys
    "vote_average":       "float32",
    "vote_count":         "int32",
    "popularity":         "float32",
    "number_of_seasons":  "int16",
    "number_of_episodes": "int32",
}

# ---------------------------------------------------------------------------
# Module-level singletons (populated once by load())
# ---------------------------------------------------------------------------
_df: Optional[pd.DataFrame] = None          # The single authoritative DataFrame
_id_to_pos: Optional[dict[int, int]] = None  # {show_id: integer position in _df}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(csv_path: Path = PROCESSED_CSV) -> None:
    """Load and validate the processed dataset. Safe to call multiple times;
    subsequent calls are no-ops unless *force=True* is passed.

    Parameters
    ----------
    csv_path:
        Path to the processed CSV file (override in tests).
    """
    global _df, _id_to_pos

    if _df is not None:
        return  # Already loaded

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {csv_path}.\n"
            "Run: .venv/bin/python scripts/preprocess.py"
        )

    _df = pd.read_csv(
        csv_path,
        usecols=REQUIRED_COLUMNS,
        dtype=_DTYPE_MAP,
        low_memory=False,
    )

    # ---- Validation ----
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in _df.columns]
    if missing_cols:
        _df = None
        raise ValueError(f"Processed CSV is missing required columns: {missing_cols}")

    if not _df["id"].is_unique:
        dup_count = int(_df["id"].duplicated().sum())
        _df = None
        raise ValueError(
            f"Processed CSV has {dup_count} duplicate IDs. "
            "Re-run scripts/preprocess.py to fix the dataset."
        )

    if _df["name"].isnull().any() or (_df["name"].str.strip() == "").any():
        bad = int((_df["name"].isnull() | (_df["name"].str.strip() == "")).sum())
        # Non-fatal: just warn; preprocessor should have removed these.
        import warnings
        warnings.warn(f"{bad} rows have missing or blank names.", stacklevel=2)

    # Safe fill for text columns that may be empty strings after preprocessing
    for col in ("overview", "genres", "recommendation_text", "poster_path", "first_air_date"):
        _df[col] = _df[col].fillna("").astype(str)

    # Build O(1) lookup: show_id (int) → integer position in DataFrame
    _id_to_pos = {int(sid): pos for pos, sid in enumerate(_df["id"])}


def reload(csv_path: Path = PROCESSED_CSV) -> None:
    """Force a fresh load, discarding the cached DataFrame and index."""
    global _df, _id_to_pos
    _df = None
    _id_to_pos = None
    load(csv_path)


def get_dataframe() -> pd.DataFrame:
    """Return the loaded DataFrame. Raises RuntimeError if load() hasn't been called."""
    if _df is None:
        raise RuntimeError("Dataset not loaded. Call src.data_loader.load() first.")
    return _df


def get_id_index() -> dict[int, int]:
    """Return the {id → position} lookup dict."""
    if _id_to_pos is None:
        raise RuntimeError("Dataset not loaded. Call src.data_loader.load() first.")
    return _id_to_pos


def get_show_by_id(show_id: int) -> Optional[dict]:
    """Return a plain dict of metadata for *show_id*, or None if not found.

    Returns a dict with all REQUIRED_COLUMNS as keys. Empty strings for missing
    text fields; 0 / 0.0 for missing numerics (handled in preprocessing).
    """
    idx = get_id_index().get(int(show_id))
    if idx is None:
        return None
    row = get_dataframe().iloc[idx]
    return _row_to_dict(row)


def search_shows(
    query: str = "",
    genre: str = "",
    sort_by: str = "popularity",
    ascending: bool = False,
    page: int = 1,
    page_size: int = 24,
    min_rating: Optional[float] = None,
    min_votes: Optional[int] = None,
) -> dict:
    """Search, filter, sort and paginate shows.

    min_rating: keep only shows with vote_average strictly greater than this value.
    min_votes:  keep only shows with at least this many votes.

    Parameters
    ----------
    query:    Case-insensitive substring match against *name* and *overview*.
    genre:    Exact-match against one of the comma-separated genre tokens.
    sort_by:  Column name to sort by; must be one of the allowed sort keys.
    ascending: Sort direction.
    page:     1-based page number.
    page_size: Rows per page (capped at 48 to keep browser memory light).

    Returns
    -------
    {"results": [dict, ...], "total": int, "page": int, "page_size": int}
    """
    _SORT_ALLOW = {"popularity", "vote_average", "vote_count", "first_air_date", "name"}
    if sort_by not in _SORT_ALLOW:
        sort_by = "popularity"

    page_size = min(int(page_size), 48)
    page = max(int(page), 1)

    df = get_dataframe()

    # --- Filter ---
    mask = pd.Series([True] * len(df), index=df.index)

    if query:
        q = query.strip().lower()
        name_match = df["name"].str.lower().str.contains(q, na=False, regex=False)
        overview_match = df["overview"].str.lower().str.contains(q, na=False, regex=False)
        mask &= name_match | overview_match

    if genre:
        g = genre.strip()
        if g:
            # Matches genre as an exact token in comma-separated list, case-insensitively.
            # e.g., "Drama" matches "Drama, Crime" or "Crime, drama", but not "Melodrama".
            pattern = r"(?:^|,\s*)" + re.escape(g) + r"(?:\s*,|$)"
            mask &= df["genres"].astype(str).str.contains(
                pattern,
                case=False,
                regex=True,
                na=False,
            )

    if min_rating is not None:
        mask &= df["vote_average"] > float(min_rating)
    if min_votes is not None:
        mask &= df["vote_count"] >= int(min_votes)

    filtered = df[mask]

    total = len(filtered)

    # --- Sort ---
    filtered = filtered.sort_values(by=sort_by, ascending=ascending)

    # --- Paginate ---
    start = (page - 1) * page_size
    end = start + page_size
    page_df = filtered.iloc[start:end]

    return {
        "results": [_row_to_dict(row) for _, row in page_df.iterrows()],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


def get_all_genres() -> list[str]:
    """Return a sorted list of unique genre tokens extracted from all shows."""
    df = get_dataframe()
    genres_series = df["genres"].dropna()
    genres_series = genres_series[genres_series != ""]

    genre_set: set[str] = set()
    for entry in genres_series:
        for token in entry.split(","):
            token = token.strip()
            if token:
                genre_set.add(token)

    return sorted(genre_set)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row: pd.Series) -> dict:
    """Convert a DataFrame row to a JSON-serialisable dict."""
    return {
        "id":                 int(row["id"]),
        "name":               str(row["name"]),
        "overview":           str(row["overview"]),
        "genres":             str(row["genres"]),
        "vote_average":       round(float(row["vote_average"]), 3),
        "vote_count":         int(row["vote_count"]),
        "popularity":         round(float(row["popularity"]), 3),
        "first_air_date":     str(row["first_air_date"]),
        "poster_path":        str(row["poster_path"]),
        "number_of_seasons":  int(row["number_of_seasons"]),
        "number_of_episodes": int(row["number_of_episodes"]),
        "recommendation_text": str(row["recommendation_text"]),
    }

