"""
tests/test_api.py

BingeBuddy - Phase 4 Flask API Route Tests
==========================================
Tests the Flask endpoints using small synthetic fixtures.
The 164k-row dataset is never loaded during these tests.

Endpoints covered:
- GET /
- GET /api/genres
- GET /api/shows
- GET /api/shows/<show_id>
- GET /api/shows/<show_id>/recommendations

Run with:
    .venv/bin/python -m unittest tests.test_api -v
"""

import json
import unittest

from app import create_app
import src.recommender as recommender
from tests.test_data_loader_and_recommender import (
    _inject_fixture,
    _make_fixture_df,
    _reset_all,
)


class TestFlaskAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _reset_all()
        # Inject small synthetic fixture (7 shows)
        _inject_fixture(_make_fixture_df())
        recommender.build()

        cls.app = create_app(init_dataset=False)
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        _reset_all()

    # -----------------------------------------------------------------------
    # Route: GET /
    # -----------------------------------------------------------------------

    def test_root_route_returns_200_html(self):
        """GET / returns HTTP 200 with HTML content."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["Content-Type"])
        self.assertIn("BingeBuddy", response.get_data(as_text=True))

    # -----------------------------------------------------------------------
    # Route: GET /api/genres
    # -----------------------------------------------------------------------

    def test_get_genres_returns_sorted_list(self):
        """GET /api/genres returns JSON with sorted genre list."""
        response = self.client.get("/api/genres")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn("genres", data)
        self.assertIn("total", data)
        self.assertIsInstance(data["genres"], list)
        self.assertGreater(data["total"], 0)
        self.assertEqual(data["genres"], sorted(data["genres"]))
        self.assertIn("Crime", data["genres"])
        self.assertIn("Drama", data["genres"])

    # -----------------------------------------------------------------------
    # Route: GET /api/shows
    # -----------------------------------------------------------------------

    def test_list_shows_default_pagination(self):
        """GET /api/shows returns paginated results structure."""
        response = self.client.get("/api/shows")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn("results", data)
        self.assertIn("total", data)
        self.assertIn("page", data)
        self.assertIn("page_size", data)
        self.assertEqual(data["page"], 1)
        self.assertEqual(data["page_size"], 24)
        self.assertEqual(len(data["results"]), 7)  # All 7 fixture shows fit on page 1

    def test_list_shows_query_search(self):
        """GET /api/shows?query=Alpha returns matched show."""
        response = self.client.get("/api/shows?query=Alpha")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["results"][0]["name"], "Alpha Crime")

    def test_list_shows_genre_filter(self):
        """GET /api/shows?genre=Food returns matching show."""
        response = self.client.get("/api/shows?genre=Food")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["results"][0]["id"], 6)

    def test_list_shows_sorting(self):
        """GET /api/shows?sort_by=vote_average&order=desc sorts correctly."""
        response = self.client.get("/api/shows?sort_by=vote_average&order=desc")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        ratings = [r["vote_average"] for r in data["results"]]
        self.assertEqual(ratings, sorted(ratings, reverse=True))

    def test_list_shows_invalid_sort_by(self):
        """GET /api/shows?sort_by=invalid returns HTTP 400 Bad Request."""
        response = self.client.get("/api/shows?sort_by=unknown_column")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_list_shows_invalid_page(self):
        """GET /api/shows?page=0 or page=abc returns HTTP 400 Bad Request."""
        res_zero = self.client.get("/api/shows?page=0")
        self.assertEqual(res_zero.status_code, 400)

        res_str = self.client.get("/api/shows?page=abc")
        self.assertEqual(res_str.status_code, 400)

    def test_list_shows_invalid_page_size(self):
        """GET /api/shows?page_size=50 or page_size=-1 returns HTTP 400 Bad Request."""
        res_too_large = self.client.get("/api/shows?page_size=50")
        self.assertEqual(res_too_large.status_code, 400)

        res_neg = self.client.get("/api/shows?page_size=-5")
        self.assertEqual(res_neg.status_code, 400)

    # -----------------------------------------------------------------------
    # Route: GET /api/shows/<show_id>
    # -----------------------------------------------------------------------

    def test_get_show_by_id_known(self):
        """GET /api/shows/1 returns HTTP 200 with complete metadata."""
        response = self.client.get("/api/shows/1")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["id"], 1)
        self.assertEqual(data["name"], "Alpha Crime")
        self.assertIn("overview", data)
        self.assertIn("genres", data)
        self.assertIn("vote_average", data)
        self.assertIn("vote_count", data)
        self.assertIn("popularity", data)

    def test_get_show_by_id_unknown(self):
        """GET /api/shows/99999 returns HTTP 404 Not Found."""
        response = self.client.get("/api/shows/99999")
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertIn("error", data)
        self.assertIn("99999", data["error"])

    # -----------------------------------------------------------------------
    # Route: GET /api/shows/<show_id>/recommendations
    # -----------------------------------------------------------------------

    def test_get_recommendations_known_positive(self):
        """GET /api/shows/1/recommendations returns top-5 similar shows."""
        response = self.client.get("/api/shows/1/recommendations")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn("show_id", data)
        self.assertIn("recommendations", data)
        self.assertEqual(data["show_id"], 1)

        rec_ids = [r["id"] for r in data["recommendations"]]
        self.assertNotIn(1, rec_ids, "Queried show 1 must never be recommended to itself.")
        self.assertNotIn(6, rec_ids, "Show 6 with 0.0 similarity must be excluded.")
        self.assertGreater(len(data["recommendations"]), 0)
        self.assertLessEqual(len(data["recommendations"]), 5)

    def test_get_recommendations_no_positive_similarity(self):
        """GET /api/shows/6/recommendations returns empty recommendations and warning."""
        response = self.client.get("/api/shows/6/recommendations")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["show_id"], 6)
        self.assertEqual(data["recommendations"], [])
        self.assertEqual(data["warning"], "No shows with positive similarity found.")

    def test_get_recommendations_empty_text_show(self):
        """GET /api/shows/7/recommendations returns empty recommendations and warning."""
        response = self.client.get("/api/shows/7/recommendations")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["show_id"], 7)
        self.assertEqual(data["recommendations"], [])
        self.assertIn("no overview or genre text", data["warning"])

    def test_get_recommendations_unknown_show(self):
        """GET /api/shows/99999/recommendations returns HTTP 404 Not Found."""
        response = self.client.get("/api/shows/99999/recommendations")
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()

