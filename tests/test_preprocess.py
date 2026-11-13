"""
tests/test_preprocess.py

Unit tests for BingeBuddy Phase 2 preprocessing pipeline:
- Exact duplicate removal
- Deterministic conflicting ID resolution
- Missing and blank title exclusion
- Recommendation text construction without fabrication
- Safe numeric clamping and missing value handling
- ID uniqueness assertion
"""

import unittest
import pandas as pd
import numpy as np
from scripts.preprocess import (
    clean_dataset,
    resolve_conflicting_ids,
    construct_recommendation_text,
)


class TestPreprocessingPipeline(unittest.TestCase):
    def setUp(self):
        """Create a synthetic dataset covering all test cases."""
        self.sample_data = pd.DataFrame([
            # 1. Normal record with full metadata
            {
                "id": 101,
                "name": "Breaking Bad",
                "overview": "A chemistry teacher turns to crime.",
                "genres": "Drama, Crime",
                "vote_average": 9.5,
                "vote_count": 12000,
                "popularity": 85.4,
                "first_air_date": "2008-01-20",
                "poster_path": "/poster101.jpg",
                "number_of_seasons": 5,
                "number_of_episodes": 62,
            },
            # 2. Exact duplicate of record #1 (should be removed)
            {
                "id": 101,
                "name": "Breaking Bad",
                "overview": "A chemistry teacher turns to crime.",
                "genres": "Drama, Crime",
                "vote_average": 9.5,
                "vote_count": 12000,
                "popularity": 85.4,
                "first_air_date": "2008-01-20",
                "poster_path": "/poster101.jpg",
                "number_of_seasons": 5,
                "number_of_episodes": 62,
            },
            # 3. Conflicting ID 202 - Less complete version
            {
                "id": 202,
                "name": "Sherlock",
                "overview": np.nan,  # Missing overview
                "genres": "Crime",
                "vote_average": 8.0,
                "vote_count": 500,
                "popularity": 20.0,
                "first_air_date": "2010-07-25",
                "poster_path": np.nan,  # Missing poster
                "number_of_seasons": 4,
                "number_of_episodes": 13,
            },
            # 4. Conflicting ID 202 - More complete version (higher completeness & vote_count)
            {
                "id": 202,
                "name": "Sherlock",
                "overview": "A modern update finds the famous sleuth in 21st-century London.",
                "genres": "Crime, Drama, Mystery",
                "vote_average": 8.4,
                "vote_count": 4500,
                "popularity": 45.0,
                "first_air_date": "2010-07-25",
                "poster_path": "/sherlock.jpg",
                "number_of_seasons": 4,
                "number_of_episodes": 13,
            },
            # 5. Record with missing title (None) -> should be excluded
            {
                "id": 303,
                "name": None,
                "overview": "Some show overview",
                "genres": "Comedy",
                "vote_average": 7.0,
                "vote_count": 10,
                "popularity": 5.0,
                "first_air_date": "2020-01-01",
                "poster_path": "/test.jpg",
                "number_of_seasons": 1,
                "number_of_episodes": 10,
            },
            # 6. Record with whitespace-only title -> should be excluded
            {
                "id": 304,
                "name": "   ",
                "overview": "Another show",
                "genres": "Drama",
                "vote_average": 6.5,
                "vote_count": 5,
                "popularity": 3.0,
                "first_air_date": "2021-01-01",
                "poster_path": "/test2.jpg",
                "number_of_seasons": 1,
                "number_of_episodes": 8,
            },
            # 7. Record with invalid/boundary numeric values (vote_average > 10, negative votes)
            {
                "id": 405,
                "name": "Wild Range",
                "overview": "Testing boundary numbers",
                "genres": "Action",
                "vote_average": 15.0,  # Out of range, should clamp to 10.0
                "vote_count": -5,      # Negative, should clamp to 0
                "popularity": -1.2,    # Negative, should clamp to 0.0
                "first_air_date": "2022-05-10",
                "poster_path": "/wild.jpg",
                "number_of_seasons": -1,
                "number_of_episodes": -10,
            },
            # 8. Record with missing overview & genres (should handle cleanly without fabricating)
            {
                "id": 506,
                "name": "Mystery Program",
                "overview": np.nan,
                "genres": np.nan,
                "vote_average": 0.0,
                "vote_count": 0,
                "popularity": 1.0,
                "first_air_date": "2023-01-01",
                "poster_path": np.nan,
                "number_of_seasons": 1,
                "number_of_episodes": 1,
            },
        ])

    def test_construct_recommendation_text(self):
        """Test text construction without fabricating content."""
        # Both present
        self.assertEqual(
            construct_recommendation_text("An epic tale.", "Fantasy, Action"),
            "Fantasy, Action. An epic tale.",
        )
        # Genres only
        self.assertEqual(
            construct_recommendation_text("", "Comedy"),
            "Comedy",
        )
        # Overview only
        self.assertEqual(
            construct_recommendation_text("Just a summary.", ""),
            "Just a summary.",
        )
        # Neither present (handles nan and None)
        self.assertEqual(construct_recommendation_text(None, None), "")
        self.assertEqual(construct_recommendation_text("nan", "nan"), "")
        self.assertEqual(construct_recommendation_text("   ", "   "), "")

    def test_exact_duplicates_removed(self):
        """Test that identical rows are removed."""
        cleaned_df, metrics = clean_dataset(self.sample_data)
        self.assertEqual(metrics["exact_duplicates_removed"], 1)

    def test_conflicting_id_resolution(self):
        """Test that conflicting IDs are resolved deterministically favoring complete records."""
        cleaned_df, metrics = clean_dataset(self.sample_data)
        # Record 202 had two conflicting versions
        self.assertEqual(metrics["unique_conflicting_ids"], 1)

        # Confirm the preserved 202 is the complete version
        sherlock = cleaned_df[cleaned_df["id"] == 202].iloc[0]
        self.assertEqual(sherlock["vote_count"], 4500)
        self.assertEqual(sherlock["poster_path"], "/sherlock.jpg")
        self.assertIn("famous sleuth", sherlock["overview"])

    def test_missing_and_blank_title_exclusion(self):
        """Test that rows with null or whitespace-only titles are excluded."""
        cleaned_df, metrics = clean_dataset(self.sample_data)
        self.assertEqual(metrics["records_removed_missing_title"], 2)
        # IDs 303 and 304 should not be in cleaned_df
        self.assertNotIn(303, cleaned_df["id"].values)
        self.assertNotIn(304, cleaned_df["id"].values)

    def test_numeric_clamping(self):
        """Test that out-of-range numeric values are properly clamped."""
        cleaned_df, _ = clean_dataset(self.sample_data)
        wild_range = cleaned_df[cleaned_df["id"] == 405].iloc[0]
        self.assertAlmostEqual(wild_range["vote_average"], 10.0, places=2)
        self.assertEqual(wild_range["vote_count"], 0)
        self.assertAlmostEqual(wild_range["popularity"], 0.0, places=2)
        self.assertEqual(wild_range["number_of_seasons"], 0)
        self.assertEqual(wild_range["number_of_episodes"], 0)

    def test_id_uniqueness(self):
        """Test that every ID in the cleaned dataset is unique."""
        cleaned_df, metrics = clean_dataset(self.sample_data)
        self.assertEqual(len(cleaned_df), metrics["unique_id_count"])
        self.assertTrue(cleaned_df["id"].is_unique)


if __name__ == "__main__":
    unittest.main()

