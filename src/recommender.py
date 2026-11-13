"""
src/recommender.py

BingeBuddy - Phase 3 TF-IDF Recommendation Engine
===================================================
Builds a sparse TF-IDF representation of all shows once and reuses it
for on-demand similarity queries. Never builds a full all-pairs matrix.

Design decisions for 4 GB RAM:
- TfidfVectorizer produces a scipy.sparse.csr_matrix (~10–30 MB for 165 k rows).
- Per-query similarity is computed with sklearn's linear_kernel against a single
  row vector, not the full NxN product. Cost per query ≈ O(N * avg_nonzeros).
- The dense NxN product (165k² × 4 bytes ≈ 109 GB) is never materialised.
- Shows with empty recommendation_text get a zero vector; they can appear as
  results for other shows but will never themselves generate good recommendations.
  The engine handles this case explicitly.
- The vectorizer and matrix are module-level singletons, built once.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import src.data_loader as data_loader

# ---------------------------------------------------------------------------
# Module-level singletons (built once by build())
# ---------------------------------------------------------------------------
_vectorizer: Optional[TfidfVectorizer] = None
_tfidf_matrix = None          # scipy.sparse.csr_matrix, shape (N, vocab)
_usable_mask: Optional[np.ndarray] = None  # bool array: True where rec text is non-empty
_usable_positions: Optional[np.ndarray] = None  # int array of positions that are usable


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build() -> None:
    """Fit the TF-IDF vectorizer on recommendation_text from the loaded dataset.

    Idempotent — safe to call multiple times; subsequent calls are no-ops.
    Must be called after src.data_loader.load().
    """
    global _vectorizer, _tfidf_matrix, _usable_mask, _usable_positions

    if _tfidf_matrix is not None:
        return  # Already built

    df = data_loader.get_dataframe()
    texts: list[str] = df["recommendation_text"].tolist()

    # Build mask of rows that have non-empty recommendation text.
    # Shows without any text get a zero TF-IDF vector and cannot produce
    # meaningful recommendations — we track them explicitly.
    _usable_mask = np.array([bool(t.strip()) for t in texts], dtype=bool)
    _usable_positions = np.where(_usable_mask)[0]

    # Fit vectorizer on ALL rows (including empty strings); sklearn handles
    # empty strings gracefully, yielding all-zero rows without error.
    # min_df is 5 as required (clamped to n_docs if n_docs < 5 to support small synthetic fixtures).
    n_docs = len(texts)
    effective_min_df = max(1, min(5, n_docs))
    _vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1),
        min_df=effective_min_df,
        max_features=40000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    _tfidf_matrix = _vectorizer.fit_transform(texts)


def rebuild() -> None:
    """Force a rebuild of the TF-IDF matrix (e.g. after data_loader.reload())."""
    global _vectorizer, _tfidf_matrix, _usable_mask, _usable_positions
    _vectorizer = None
    _tfidf_matrix = None
    _usable_mask = None
    _usable_positions = None
    build()


def get_recommendations(
    show_id: int,
    top_n: int = 5,
) -> dict:
    """Return the *top_n* most similar shows for *show_id*.

    Parameters
    ----------
    show_id : int
        TMDB show ID to base recommendations on.
    top_n : int
        Number of recommendations to return (default 5).

    Returns
    -------
    dict with keys:
        "show_id"          : the requested ID (int)
        "recommendations"  : list of show dicts (may be shorter than top_n if
                             the dataset is small or the show has no usable text)
        "warning"          : optional human-readable string describing any
                             degraded-mode reason (empty string if all is well)
    """
    if _tfidf_matrix is None:
        raise RuntimeError(
            "Recommender not built. Call src.recommender.build() first."
        )

    id_index = data_loader.get_id_index()
    df = data_loader.get_dataframe()

    # --- Validate show_id ---
    if int(show_id) not in id_index:
        return {
            "show_id": show_id,
            "recommendations": [],
            "warning": f"Show ID {show_id} not found in the dataset.",
        }

    query_pos = id_index[int(show_id)]
    warning = ""

    # --- Check usable text ---
    if not _usable_mask[query_pos]:
        warning = (
            f"Show ID {show_id} has no overview or genre text. "
            "Recommendations cannot be generated from content alone."
        )
        return {
            "show_id": show_id,
            "recommendations": [],
            "warning": warning,
        }

    # --- Check minimum dataset size ---
    # Need at least top_n + 1 usable shows (excluding the query itself)
    n_usable = int(_usable_mask.sum())
    if n_usable < 2:
        return {
            "show_id": show_id,
            "recommendations": [],
            "warning": "Dataset has fewer than 2 shows with usable text; cannot recommend.",
        }

    # --- Compute similarity on demand (sparse, single-row dot product) ---
    # query_vec: shape (1, vocab) sparse
    query_vec = _tfidf_matrix[query_pos]

    # linear_kernel computes dot products between query_vec and all rows.
    # Returns a (1, N) ndarray. We immediately flatten to avoid holding (N, N).
    sim_scores: np.ndarray = linear_kernel(query_vec, _tfidf_matrix).flatten()

    # Exclude the query show itself by setting its score to -1
    sim_scores[query_pos] = -1.0

    # Also zero-out shows with no recommendation text so they are never surfaced
    sim_scores[~_usable_mask] = -1.0

    # Exclude candidates whose similarity score is zero or negative
    positive_indices = np.where(sim_scores > 0.0)[0]
    n_positive = len(positive_indices)

    if n_positive == 0:
        return {
            "show_id": int(show_id),
            "recommendations": [],
            "warning": "No shows with positive similarity found.",
        }

    n_candidates = min(top_n, n_positive)
    if n_positive <= n_candidates:
        # Sort all positive candidates descending by score
        sorted_sub = np.argsort(sim_scores[positive_indices])[::-1]
        top_indices = positive_indices[sorted_sub]
    else:
        # Partition to find top n_candidates without sorting all rows
        pos_scores = sim_scores[positive_indices]
        partitioned = np.argpartition(pos_scores, -n_candidates)[-n_candidates:]
        sorted_top_sub = partitioned[np.argsort(pos_scores[partitioned])[::-1]]
        top_indices = positive_indices[sorted_top_sub]

    # Build result list from DataFrame rows
    results = []
    for idx in top_indices:
        row = df.iloc[int(idx)]
        results.append(data_loader._row_to_dict(row))

    return {
        "show_id": int(show_id),
        "recommendations": results,
        "warning": warning,
    }


def is_built() -> bool:
    """Return True if the TF-IDF matrix has been built."""
    return _tfidf_matrix is not None


def matrix_info() -> dict:
    """Return diagnostic info about the current TF-IDF matrix."""
    if _tfidf_matrix is None:
        return {"built": False}
    return {
        "built": True,
        "shape": list(_tfidf_matrix.shape),
        "stored_elements": int(_tfidf_matrix.nnz),
        "usable_shows": int(_usable_mask.sum()) if _usable_mask is not None else 0,
        "vocabulary_size": len(_vectorizer.vocabulary_) if _vectorizer else 0,
    }
