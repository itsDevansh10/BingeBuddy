"""
tests/test_data_loader_and_recommender.py

BingeBuddy - Phase 3 Unit Tests
================================
Tests for src/data_loader.py and src/recommender.py using small synthetic fixtures.
The full 164k-row dataset is never loaded in these tests.

Conventions match test_preprocess.py: plain unittest, no extra test runner.

Run with:
    .venv/bin/python -m unittest tests.test_data_loader_and_recommender -v
"""

import io
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import src.data_loader as data_loader
import src.recommender as recommender


# ---------------------------------------------------------------------------
# Shared fixture factory
# ---------------------------------------------------------------------------

def _make_fixture_df(rows: list[dict] | None = None) -> pd.DataFrame:
    """Return a small synthetic DataFrame shaped like shows.csv.
    Default fixture has terms ('crime', 'drama', 'detectives', 'city', 'series')
    appearing in 5 shows so that min_df=5 produces a valid non-empty vocabulary.
    """
    if rows is None:
        rows = [
            {
                "id": 1, "name": "Alpha Crime",
                "overview": "Detectives solve crime in city series.",
                "genres": "Crime, Drama",
                "vote_average": 8.5, "vote_count": 1000, "popularity": 90.0,
                "first_air_date": "2010-01-01", "poster_path": "/alpha.jpg",
                "number_of_seasons": 3, "number_of_episodes": 30,
                "recommendation_text": "Crime, Drama. Detectives solve crime in city series.",
            },
            {
                "id": 2, "name": "Beta Crime",
                "overview": "Detectives chase criminals across dark city series.",
                "genres": "Crime, Drama",
                "vote_average": 7.8, "vote_count": 500, "popularity": 60.0,
                "first_air_date": "2012-06-15", "poster_path": "/beta.jpg",
                "number_of_seasons": 2, "number_of_episodes": 20,
                "recommendation_text": "Crime, Drama. Detectives chase criminals across dark city series.",
            },
            {
                "id": 3, "name": "Gamma Investigation",
                "overview": "Police detectives investigate city crime series.",
                "genres": "Crime, Drama",
                "vote_average": 6.9, "vote_count": 200, "popularity": 30.0,
                "first_air_date": "2015-09-01", "poster_path": "/gamma.jpg",
                "number_of_seasons": 5, "number_of_episodes": 100,
                "recommendation_text": "Crime, Drama. Police detectives investigate city crime series.",
            },
            {
                "id": 4, "name": "Delta Precinct",
                "overview": "Detectives work crime in dark city series.",
                "genres": "Crime, Drama",
                "vote_average": 9.0, "vote_count": 3000, "popularity": 120.0,
                "first_air_date": "2018-03-10", "poster_path": "/delta.jpg",
                "number_of_seasons": 1, "number_of_episodes": 10,
                "recommendation_text": "Crime, Drama. Detectives work crime in dark city series.",
            },
            {
                "id": 5, "name": "Epsilon Mystery",
                "overview": "Detectives hunt crime in city series.",
                "genres": "Crime, Drama",
                "vote_average": 8.1, "vote_count": 1500, "popularity": 80.0,
                "first_air_date": "2020-07-20", "poster_path": "/epsilon.jpg",
                "number_of_seasons": 2, "number_of_episodes": 16,
                "recommendation_text": "Crime, Drama. Detectives hunt crime in city series.",
            },
            {
                "id": 6, "name": "Zeta Cooking",
                "overview": "Chefs bake bread in a cozy kitchen.",
                "genres": "Food, Cooking",
                "vote_average": 7.0, "vote_count": 100, "popularity": 15.0,
                "first_air_date": "2021-01-01", "poster_path": "/zeta.jpg",
                "number_of_seasons": 1, "number_of_episodes": 8,
                "recommendation_text": "Food, Cooking. Chefs bake bread in a cozy kitchen.",
            },
            {
                "id": 7, "name": "Eta Empty",
                "overview": "",
                "genres": "",
                "vote_average": 0.0, "vote_count": 0, "popularity": 1.0,
                "first_air_date": "", "poster_path": "",
                "number_of_seasons": 0, "number_of_episodes": 0,
                "recommendation_text": "",
            },
        ]
    return pd.DataFrame(rows).astype({
        "id": "int64", "vote_average": "float32", "vote_count": "int32",
        "popularity": "float32", "number_of_seasons": "int16", "number_of_episodes": "int32",
    })


def _inject_fixture(df: pd.DataFrame) -> None:
    """Inject a synthetic DataFrame into data_loader's module-level singletons."""
    data_loader._df = df.copy()
    data_loader._id_to_pos = {int(sid): pos for pos, sid in enumerate(df["id"])}


def _reset_all() -> None:
    """Reset both module singletons between tests."""
    data_loader._df = None
    data_loader._id_to_pos = None
    recommender._vectorizer = None
    recommender._tfidf_matrix = None
    recommender._usable_mask = None
    recommender._usable_positions = None


# ---------------------------------------------------------------------------
# Data Loader Tests
# ---------------------------------------------------------------------------

class TestDataLoaderValidation(unittest.TestCase):

    def setUp(self):
        _reset_all()

    def tearDown(self):
        _reset_all()

    def test_get_dataframe_raises_before_load(self):
        """get_dataframe() must raise RuntimeError if load() hasn't been called."""
        with self.assertRaises(RuntimeError):
            data_loader.get_dataframe()

    def test_get_id_index_raises_before_load(self):
        """get_id_index() must raise RuntimeError if load() hasn't been called."""
        with self.assertRaises(RuntimeError):
            data_loader.get_id_index()

    def test_load_detects_missing_columns(self):
        """load() must raise ValueError when required columns are absent."""
        bad_csv = io.StringIO("id,name\n1,Alpha\n")
        df = pd.read_csv(bad_csv)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            tmp_path = Path(f.name)
        try:
            with self.assertRaises(ValueError, msg="Should raise for missing columns"):
                data_loader.load(csv_path=tmp_path)
        finally:
            os.unlink(tmp_path)
            _reset_all()

    def test_load_detects_duplicate_ids(self):
        """load() must raise ValueError when the CSV contains duplicate IDs."""
        df = _make_fixture_df()
        df = pd.concat([df, df.head(1)], ignore_index=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            tmp_path = Path(f.name)
        try:
            with self.assertRaises(ValueError, msg="Should raise for duplicate IDs"):
                data_loader.load(csv_path=tmp_path)
        finally:
            os.unlink(tmp_path)
            _reset_all()

    def test_get_show_by_id_known(self):
        """get_show_by_id() returns correct data for a known ID."""
        _inject_fixture(_make_fixture_df())
        show = data_loader.get_show_by_id(1)
        self.assertIsNotNone(show)
        self.assertEqual(show["name"], "Alpha Crime")
        self.assertEqual(show["genres"], "Crime, Drama")

    def test_get_show_by_id_unknown(self):
        """get_show_by_id() returns None for an unknown ID."""
        _inject_fixture(_make_fixture_df())
        result = data_loader.get_show_by_id(99999)
        self.assertIsNone(result)

    def test_get_all_genres_non_empty(self):
        """get_all_genres() returns a non-empty sorted list."""
        _inject_fixture(_make_fixture_df())
        genres = data_loader.get_all_genres()
        self.assertGreater(len(genres), 0)
        self.assertEqual(genres, sorted(genres))
        self.assertIn("Crime", genres)
        self.assertIn("Cooking", genres)

    def test_row_to_dict_serialisable_types(self):
        """_row_to_dict() must return plain Python types (JSON-serialisable)."""
        _inject_fixture(_make_fixture_df())
        show = data_loader.get_show_by_id(1)
        self.assertIsInstance(show["id"], int)
        self.assertIsInstance(show["vote_average"], float)
        self.assertIsInstance(show["vote_count"], int)
        self.assertIsInstance(show["popularity"], float)
        self.assertIsInstance(show["name"], str)

    def test_missing_text_fields_return_empty_string(self):
        """Show with missing overview/genres/poster must return empty strings, not NaN."""
        _inject_fixture(_make_fixture_df())
        show = data_loader.get_show_by_id(7)
        self.assertIsNotNone(show)
        self.assertEqual(show["overview"], "")
        self.assertEqual(show["genres"], "")
        self.assertEqual(show["poster_path"], "")

    def test_genre_filter_exact_token_case_insensitive(self):
        """Genre filtering must match exact comma-separated tokens, case-insensitively."""
        _inject_fixture(_make_fixture_df())

        # Exact match (Crime)
        res_crime = data_loader.search_shows(genre="Crime")
        self.assertEqual(res_crime["total"], 5)
        for r in res_crime["results"]:
            self.assertIn("Crime", r["genres"])

        # Case-insensitive (crime, DRAMA)
        res_lower = data_loader.search_shows(genre="crime")
        self.assertEqual(res_lower["total"], 5)

        res_upper = data_loader.search_shows(genre="DRAMA")
        self.assertEqual(res_upper["total"], 5)

        # Single distinct genre (Food)
        res_food = data_loader.search_shows(genre="Food")
        self.assertEqual(res_food["total"], 1)
        self.assertEqual(res_food["results"][0]["id"], 6)

        # Partial substring should not match full token
        res_partial = data_loader.search_shows(genre="Cook")
        self.assertEqual(res_partial["total"], 0)

        # Nonexistent genre
        res_none = data_loader.search_shows(genre="NonExistentGenre")
        self.assertEqual(res_none["total"], 0)

    def test_search_shows_pagination_and_sorting(self):
        """search_shows pagination and sorting work as expected."""
        _inject_fixture(_make_fixture_df())

        # Sort by popularity descending
        res_pop = data_loader.search_shows(sort_by="popularity", ascending=False, page_size=2)
        self.assertEqual(res_pop["page"], 1)
        self.assertEqual(res_pop["page_size"], 2)
        self.assertEqual(len(res_pop["results"]), 2)
        self.assertGreaterEqual(res_pop["results"][0]["popularity"], res_pop["results"][1]["popularity"])


# ---------------------------------------------------------------------------
# Recommender Tests
# ---------------------------------------------------------------------------

class TestRecommender(unittest.TestCase):

    def setUp(self):
        _reset_all()
        _inject_fixture(_make_fixture_df())
        recommender.build()

    def tearDown(self):
        _reset_all()

    # ---- Build & Configuration checks ----

    def test_tfidf_vectorizer_configuration(self):
        """TfidfVectorizer must be configured with ngram_range=(1, 1), min_df=5, max_features=40000."""
        self.assertEqual(recommender._vectorizer.ngram_range, (1, 1))
        self.assertEqual(recommender._vectorizer.min_df, 5)
        self.assertEqual(recommender._vectorizer.max_features, 40000)

    def test_matrix_is_sparse(self):
        """TF-IDF matrix must be a sparse CSR matrix (never dense)."""
        import scipy.sparse
        self.assertTrue(scipy.sparse.issparse(recommender._tfidf_matrix))
        self.assertTrue(scipy.sparse.isspmatrix_csr(recommender._tfidf_matrix))

    def test_is_built_after_build(self):
        """is_built() must return True after build()."""
        self.assertTrue(recommender.is_built())

    def test_build_is_idempotent(self):
        """Calling build() twice must not replace the existing matrix."""
        first_matrix = recommender._tfidf_matrix
        recommender.build()
        self.assertIs(recommender._tfidf_matrix, first_matrix)

    def test_matrix_info_reports_usable_shows(self):
        """matrix_info() must correctly count shows with non-empty rec text."""
        info = recommender.matrix_info()
        self.assertTrue(info["built"])
        # 6 out of 7 fixture shows have non-empty recommendation_text
        self.assertEqual(info["usable_shows"], 6)

    # ---- Core correctness & Similarity filtering ----

    def test_show_never_recommends_itself(self):
        """The queried show must never appear in its own recommendations."""
        for show_id in [1, 2, 3, 4, 5]:
            result = recommender.get_recommendations(show_id)
            rec_ids = [r["id"] for r in result["recommendations"]]
            self.assertNotIn(show_id, rec_ids, f"Show {show_id} appeared in its own recs.")

    def test_recommendation_ids_are_unique(self):
        """No duplicate show IDs in a single recommendation list."""
        result = recommender.get_recommendations(1)
        rec_ids = [r["id"] for r in result["recommendations"]]
        self.assertEqual(len(rec_ids), len(set(rec_ids)))

    def test_zero_similarity_candidates_are_excluded(self):
        """Candidates with 0.0 similarity (e.g. Zeta Cooking for Show 1) must be excluded."""
        result = recommender.get_recommendations(1, top_n=10)
        rec_ids = [r["id"] for r in result["recommendations"]]
        # Shows 2, 3, 4, 5 share vocabulary terms; Show 6 (Cooking) has 0.0 similarity
        self.assertNotIn(6, rec_ids, "Show 6 with 0.0 similarity must not be included.")
        # Ensure only positive-similarity shows are returned
        self.assertGreater(len(rec_ids), 0)
        self.assertTrue(set(rec_ids).issubset({2, 3, 4, 5}))

    def test_no_positive_similarity_returns_empty_with_warning(self):
        """When no candidate has positive similarity, return empty list and clear warning."""
        # Show 6 shares no vocabulary terms with any other show
        result = recommender.get_recommendations(6)
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["warning"], "No shows with positive similarity found.")

    def test_returned_metadata_fields(self):
        """Each recommendation dict must contain all required metadata keys."""
        required_keys = {
            "id", "name", "overview", "genres", "vote_average",
            "vote_count", "popularity", "first_air_date", "poster_path",
            "number_of_seasons", "number_of_episodes", "recommendation_text",
        }
        result = recommender.get_recommendations(1)
        for rec in result["recommendations"]:
            self.assertTrue(required_keys.issubset(rec.keys()),
                            f"Missing keys: {required_keys - rec.keys()}")

    # ---- Edge cases ----

    def test_unknown_show_id_returns_empty_with_warning(self):
        """Unknown show ID must return empty recommendations and a warning string."""
        result = recommender.get_recommendations(99999)
        self.assertEqual(result["recommendations"], [])
        self.assertIn("99999", result["warning"])

    def test_show_with_empty_text_returns_empty_with_warning(self):
        """Show with no overview/genres (ID=7) cannot generate recommendations."""
        result = recommender.get_recommendations(7)
        self.assertEqual(result["recommendations"], [])
        self.assertIn("no overview or genre text", result["warning"])

    def test_tiny_dataset_fewer_than_6_shows(self):
        """Dataset with 2 usable shows returns at most 1 recommendation (not top_n=5)."""
        _reset_all()
        tiny_df = _make_fixture_df([
            {
                "id": 10, "name": "Show A",
                "overview": "Mystery thriller set at sea.",
                "genres": "Thriller",
                "vote_average": 7.0, "vote_count": 100, "popularity": 10.0,
                "first_air_date": "2020-01-01", "poster_path": "/a.jpg",
                "number_of_seasons": 1, "number_of_episodes": 6,
                "recommendation_text": "Thriller. Mystery thriller set at sea.",
            },
            {
                "id": 11, "name": "Show B",
                "overview": "A psychological thriller on a cruise ship.",
                "genres": "Thriller",
                "vote_average": 6.5, "vote_count": 50, "popularity": 5.0,
                "first_air_date": "2021-03-01", "poster_path": "/b.jpg",
                "number_of_seasons": 1, "number_of_episodes": 4,
                "recommendation_text": "Thriller. A psychological thriller on a cruise ship.",
            },
        ])
        _inject_fixture(tiny_df)
        recommender.build()

        result = recommender.get_recommendations(10, top_n=5)
        # Only 2 shows total; excluding self → at most 1 result
        self.assertLessEqual(len(result["recommendations"]), 1)
        rec_ids = [r["id"] for r in result["recommendations"]]
        self.assertNotIn(10, rec_ids)
        if rec_ids:
            self.assertEqual(rec_ids, [11])

    def test_not_built_raises_runtime_error(self):
        """get_recommendations() must raise RuntimeError if build() was never called."""
        _reset_all()
        _inject_fixture(_make_fixture_df())
        with self.assertRaises(RuntimeError):
            recommender.get_recommendations(1)


if __name__ == "__main__":
    unittest.main()
