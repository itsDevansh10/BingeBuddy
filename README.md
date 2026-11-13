# BingeBuddy 🎬

**BingeBuddy** is a lightweight, cinematic TV show discovery and recommendation engine. Built with Python, Flask, Pandas, and scikit-learn, it provides content-based recommendations using TF-IDF vector similarity and a responsive streaming-style user interface—engineered to run smoothly even on resource-constrained hardware (e.g. 4 GB RAM).

---

## ✨ Features

- **Content-Based Recommendations:** Powered by `scikit-learn`'s `TfidfVectorizer` and sparse cosine similarity (`linear_kernel`), computing recommendations on-demand for requested titles without memory-heavy all-pairs matrices.
- **Cinematic Discovery UI:** Responsive dark-mode interface inspired by modern streaming platforms, featuring:
  - Hero spotlight banner with backdrop artwork
  - Dynamic search with debouncing
  - Multi-genre filtering & flexible sorting (Popularity, Rating, Release Date, Title)
  - Detail modal with aggregate season/episode counts and "More Like This" recommendations
  - Safe SVG fallback placeholders for missing artwork
- **Memory-Efficient Architecture:** Custom chunked preprocessing and lazy evaluation designed to handle 160,000+ TV show records on machines with limited RAM.
- **Robust REST API:** Clean Flask endpoints for shows, genre aggregations, pagination, and recommendations.
- **Fully Tested:** Comprehensive unit tests covering data loading, text preprocessing, recommendation logic, and API routes.

---

## 🏗️ Project Structure

```text
BingeBuddy/
├── app.py                     # Flask application & REST API routes
├── requirements.txt           # Python dependencies
├── src/
│   ├── data_loader.py         # Memory-conscious dataset loader & filter
│   └── recommender.py         # Sparse TF-IDF recommender engine
├── scripts/
│   ├── inspect_dataset.py     # Safe chunked CSV inspection tool
│   ├── preprocess.py          # Data cleaning & recommendation text generator
│   └── curate.py              # Optional catalog curation filter
├── static/
│   ├── css/style.css          # Dark cinematic styling & responsive layout
│   ├── js/app.js              # Vanilla JavaScript UI logic & API client
│   └── img/
│       └── poster_placeholder.svg # Local SVG fallback poster
├── templates/
│   └── index.html             # Single-page discovery template
├── tests/
│   ├── test_data_loader_and_recommender.py
│   ├── test_preprocess.py
│   └── test_api.py
└── data/
    ├── raw/                   # Raw TMDB dataset (not tracked in git)
    └── processed/             # Cleaned dataset (shows.csv, not tracked in git)
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/itsDevansh10/BingeBuddy.git
cd BingeBuddy
```

### 3. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Prepare Dataset
1. Download the [TMDB TV Shows Dataset](https://www.kaggle.com/datasets/asaniczka/tmdb-tv-shows-dataset-2023-900k-shows) from Kaggle.
2. Place the CSV file into `data/raw/TMDB_tv_dataset_v3.csv`.
3. Run the preprocessing script:
   ```bash
   python scripts/preprocess.py
   ```
   This generates `data/processed/shows.csv` with unique records and prepared recommendation text.

*(Optional)* Build a curated catalog of widely rated shows for faster startup:
```bash
python scripts/curate.py
```

### 5. Run the Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## 🧪 Running Tests

The test suite runs using isolated synthetic fixtures and does not require the full 160k-row CSV to be loaded:

```bash
python -m unittest discover -s tests -v
```

All 46 tests should pass with zero failures.

---

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the web interface |
| `/api/genres` | `GET` | Returns list of all unique TV genres |
| `/api/shows` | `GET` | Query, filter by genre, sort, and paginate shows |
| `/api/shows/<id>` | `GET` | Returns metadata for a single show by TMDB ID |
| `/api/shows/<id>/recommendations` | `GET` | Returns top-5 content-based recommendations |

---

## ⚖️ License & Attribution

- **Dataset Attribution:** TV show metadata is provided by TMDB via Kaggle community archives. This product uses TMDB data but is not endorsed or certified by TMDB.
- **Poster Artwork:** References TMDB's public image CDN paths where available, falling back to local SVGs if offline.
