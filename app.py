"""
app.py

BingeBuddy - Flask Application & REST API
==========================================
Provides the backend REST API for TV show discovery and content-based recommendations.

Routes:
- GET /                              : Serves frontend template or minimal placeholder
- GET /api/shows                     : Search, genre filtering, sorting, pagination
- GET /api/shows/<int:show_id>       : Single show metadata lookup (or 404)
- GET /api/shows/<int:show_id>/recommendations : Top-5 similar shows via TF-IDF (or 404/empty)
- GET /api/genres                    : List of all available genre categories

Design notes:
- Uses in-memory DataFrame and sparse TF-IDF matrix loaded once per process.
- No database, no login, no remote API calls, no full pairwise matrix.
- Memory safe for 4 GB RAM laptops.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request

import src.data_loader as data_loader
import src.recommender as recommender

# Allowed sort columns to prevent invalid queries
ALLOWED_SORT_COLUMNS = {
    "popularity",
    "vote_average",
    "vote_count",
    "first_air_date",
    "name",
}


def create_app(init_dataset: bool = True) -> Flask:
    """Application factory for BingeBuddy Flask app.

    Parameters
    ----------
    init_dataset : bool
        If True, loads data_loader and builds the TF-IDF matrix once at startup.
        Set to False during unit tests to inject synthetic fixtures instead.
    """
    app = Flask(__name__)

    if init_dataset:
        try:
            data_loader.load()
            recommender.build()
        except FileNotFoundError as err:
            app.logger.warning(
                "Processed dataset not found during startup. "
                "Run 'scripts/preprocess.py' to generate 'data/processed/shows.csv'. (%s)",
                err,
            )
        except Exception as err:
            app.logger.error("Failed to initialize dataset: %s", err)

    # -----------------------------------------------------------------------
    # Route: Root / Discovery Page
    # -----------------------------------------------------------------------
    @app.route("/", methods=["GET"])
    def index():
        """Render templates/index.html if present; otherwise return a minimal placeholder."""
        template_file = Path(app.root_path) / "templates" / "index.html"
        if template_file.is_file():
            return render_template("index.html")

        placeholder_html = (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "  <meta charset=\"UTF-8\">\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            "  <title>BingeBuddy</title>\n"
            "</head>\n"
            "<body>\n"
            "  <h1>BingeBuddy API</h1>\n"
            "  <p>Discovery page placeholder. Frontend templates have not been created yet.</p>\n"
            "</body>\n"
            "</html>\n"
        )
        return placeholder_html, 200, {"Content-Type": "text/html; charset=utf-8"}

    # -----------------------------------------------------------------------
    # Route: List, Search, Filter, Sort & Paginate Shows
    # -----------------------------------------------------------------------
    @app.route("/api/shows", methods=["GET"])
    def list_shows():
        """Search, filter, sort and paginate shows.

        Query parameters:
        - query (or q) : text substring search across title and overview
        - genre        : exact token match in comma-separated genre list (case-insensitive)
        - sort_by      : sort column (popularity, vote_average, vote_count, first_air_date, name)
        - order        : 'asc' or 'desc' (default: 'desc')
        - ascending    : 'true'/'false' boolean alternative to order
        - page         : 1-based page index (default: 1)
        - page_size    : results per page between 1 and 48 (default: 24)
        - min_rating   : optional, only shows rated strictly above this value (0-10)
        - min_votes    : optional, only shows with at least this many votes
        """
        # Search query
        query = (request.args.get("query") or request.args.get("q") or "").strip()

        # Genre filter
        genre = request.args.get("genre", "").strip()

        # Sort column
        sort_by = request.args.get("sort_by", "popularity").strip().lower()
        if sort_by not in ALLOWED_SORT_COLUMNS:
            return (
                jsonify({
                    "error": (
                        f"Invalid sort_by '{sort_by}'. "
                        f"Allowed fields: {sorted(ALLOWED_SORT_COLUMNS)}"
                    )
                }),
                400,
            )

        # Sort order
        ascending_param = request.args.get("ascending")
        if ascending_param is not None:
            ascending = ascending_param.strip().lower() in ("true", "1", "yes")
        else:
            order_param = request.args.get("order", "desc").strip().lower()
            ascending = order_param == "asc"

        # Page index validation
        raw_page = request.args.get("page", 1)
        try:
            page = int(raw_page)
            if page < 1:
                return jsonify({"error": "page must be an integer >= 1."}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "page must be a valid integer."}), 400

        # Page size validation
        raw_page_size = request.args.get("page_size", 24)
        try:
            page_size = int(raw_page_size)
            if page_size < 1 or page_size > 48:
                return jsonify({"error": "page_size must be an integer between 1 and 48."}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "page_size must be a valid integer."}), 400

        # Optional quality filters (only forwarded when supplied)
        extra_filters = {}
        raw_min_rating = request.args.get("min_rating")
        if raw_min_rating not in (None, ""):
            try:
                min_rating = float(raw_min_rating)
            except ValueError:
                return jsonify({"error": "min_rating must be a number between 0 and 10."}), 400
            if not (0.0 <= min_rating <= 10.0):
                return jsonify({"error": "min_rating must be a number between 0 and 10."}), 400
            extra_filters["min_rating"] = min_rating

        raw_min_votes = request.args.get("min_votes")
        if raw_min_votes not in (None, ""):
            try:
                min_votes = int(raw_min_votes)
            except ValueError:
                return jsonify({"error": "min_votes must be a non-negative integer."}), 400
            if min_votes < 0:
                return jsonify({"error": "min_votes must be a non-negative integer."}), 400
            extra_filters["min_votes"] = min_votes

        try:
            results = data_loader.search_shows(
                query=query,
                genre=genre,
                sort_by=sort_by,
                ascending=ascending,
                page=page,
                page_size=page_size,
                **extra_filters,
            )
            return jsonify(results), 200
        except RuntimeError as err:
            return jsonify({"error": str(err)}), 503

    # -----------------------------------------------------------------------
    # Route: Single Show Metadata Lookup
    # -----------------------------------------------------------------------
    @app.route("/api/shows/<int:show_id>", methods=["GET"])
    def get_show(show_id: int):
        """Return genuine show metadata or HTTP 404 if not found."""
        try:
            show = data_loader.get_show_by_id(show_id)
        except RuntimeError as err:
            return jsonify({"error": str(err)}), 503

        if show is None:
            return jsonify({"error": f"Show ID {show_id} not found."}), 404

        return jsonify(show), 200

    # -----------------------------------------------------------------------
    # Route: Recommendations for Show
    # -----------------------------------------------------------------------
    @app.route("/api/shows/<int:show_id>/recommendations", methods=["GET"])
    def get_show_recommendations(show_id: int):
        """Return up to five similar shows, excluding the requested show."""
        try:
            show = data_loader.get_show_by_id(show_id)
        except RuntimeError as err:
            return jsonify({"error": str(err)}), 503

        if show is None:
            return jsonify({"error": f"Show ID {show_id} not found."}), 404

        try:
            rec_result = recommender.get_recommendations(show_id, top_n=5)
            return jsonify(rec_result), 200
        except RuntimeError as err:
            return jsonify({"error": str(err)}), 503

    # -----------------------------------------------------------------------
    # Route: List Available Genres
    # -----------------------------------------------------------------------
    @app.route("/api/genres", methods=["GET"])
    def list_genres():
        """Return sorted list of all unique genre tokens from dataset."""
        try:
            genres = data_loader.get_all_genres()
            return jsonify({"genres": genres, "total": len(genres)}), 200
        except RuntimeError as err:
            return jsonify({"error": str(err)}), 503

    return app


# Module-level default application instance
# Auto-initializes dataset only when not running under unit tests
_is_test_env = "unittest" in sys.modules or os.environ.get("FLASK_ENV") == "testing"
app = create_app(init_dataset=not _is_test_env)


if __name__ == "__main__":
    # When run directly, ensure dataset is initialized
    if not data_loader._df:
        data_loader.load()
    if not recommender.is_built():
        recommender.build()
    app.run(host="127.0.0.1", port=5000, debug=True)

