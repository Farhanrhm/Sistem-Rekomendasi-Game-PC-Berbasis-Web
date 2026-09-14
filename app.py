"""LevelFind: PC Game Recommendation Web Application Backend.

This module provides the Flask backend, RESTful API endpoints, in-memory model
loaders, content-based recommendation logic using TF-IDF and Cosine Similarity,
ranking optimizations (tie-breaking with review counts & sparse penalties),
diversification via Levenshtein distance, and Explainable AI (XAI) extraction.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from functools import lru_cache
import html
import os
import pickle
import re
import time
import zlib

from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import Levenshtein
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# ==============================================================================
# 1. PENGAMANAN RATE LIMITING, CORS, & SECURITY HEADERS
# Untuk production (terutama serverless seperti Vercel), set RATELIMIT_STORAGE_URI
# ke Redis (contoh: Upstash Redis URI) lewat Environment Variable.
# Default fallback ke memory:// untuk pengembangan lokal.
# ==============================================================================
RATELIMIT_STORAGE_URI: str = os.environ.get("RATELIMIT_STORAGE_URI", "memory://")

limiter: Limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["300 per day", "100 per hour"],
    storage_uri=RATELIMIT_STORAGE_URI
)

CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5000", "http://127.0.0.1:5000", "https://levelfind.vercel.app"]}})


@app.after_request
def set_secure_headers(response: Response) -> Response:
    """Attach baseline HTTP security headers and static caching rules to responses.

    Args:
        response (Response): Outgoing Flask HTTP response object.

    Returns:
        Response: Mutated HTTP response equipped with standard security headers.
    """
    # Prevent MIME-sniffing exploits
    response.headers['X-Content-Type-Options'] = 'nosniff'
    # Prevent framing to mitigate clickjacking attacks
    response.headers['X-Frame-Options'] = 'DENY'
    # Restrict referrer leakage to cross-origin requests
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    # Set public caching for static assets (1 week)
    if request.path.startswith('/static/'):
        response.headers['Cache-Control'] = 'public, max-age=604800'
    return response


def sanitize_input(user_input: Optional[str]) -> str:
    """Sanitize user search queries to prevent Cross-Site Scripting (XSS) and injection.

    Uses iterative regex stripping to eliminate nested or malformed HTML tags.

    Args:
        user_input (Optional[str]): Raw input string received from query parameters or form data.

    Returns:
        str: Cleaned, unescaped, and length-capped text query.
    """
    if not user_input or not isinstance(user_input, str):
        return ""
    clean_str = html.unescape(user_input)
    prev = None
    # Iteratively remove tags to neutralize nested injection vectors (e.g. <<SCRIPT>script>)
    while prev != clean_str:
        prev = clean_str
        clean_str = re.sub(r'<[^>]*>', '', clean_str)
    return clean_str.strip()[:100]


# ==============================================================================
# 2. BATASAN PEMODELAN & MEMORI & 3. LAZY LOADING
# PERINGATAN KEAMANAN (SECURITY NOTE):
# `pickle.load()` dapat mengeksekusi kode arbitrary jika file .pkl berasal dari sumber
# yang tidak terpercaya. Seluruh file di folder `models/` HARUS selalu diproduksi dari
# proses build internal milik developer sendiri (bukan dari unggahan publik/pihak ketiga).
# ==============================================================================
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))


def load_model_artifacts(
    base_dir: str = BASE_DIR
) -> Tuple[Optional[pd.DataFrame], Optional[csr_matrix], Optional[TfidfVectorizer], Optional[pd.Series]]:
    """Load pre-trained recommendation data artifacts from disk into RAM.

    Args:
        base_dir (str, optional): Root directory containing the models folder. Defaults to BASE_DIR.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[csr_matrix], Optional[TfidfVectorizer], Optional[pd.Series]]:
            Loaded DataFrame, Sparse TF-IDF CSR Matrix, fitted Vectorizer, and Title-to-Index Series.
    """
    print("[LAZY LOADING] Memuat file model .pkl ke dalam RAM peladen...")
    loaded_df = None
    loaded_matrix = None
    loaded_vec = None
    loaded_indices = None

    try:
        with open(os.path.join(base_dir, 'models', 'game_data.pkl'), 'rb') as f:
            loaded_df = pickle.load(f)
        with open(os.path.join(base_dir, 'models', 'tfidf_matrix.pkl'), 'rb') as f:
            loaded_matrix = pickle.load(f)
        with open(os.path.join(base_dir, 'models', 'tfidf_vectorizer.pkl'), 'rb') as f:
            loaded_vec = pickle.load(f)
        with open(os.path.join(base_dir, 'models', 'indices.pkl'), 'rb') as f:
            loaded_indices = pickle.load(f)
        print("[OK] Model dan data sparse TF-IDF berhasil dimuat ke RAM peladen!")
    except Exception as e:
        print(f"[ERROR] Error memuat file model .pkl: {e}")
        print("Harap pastikan Anda sudah menjalankan build_model.py terlebih dahulu.")

    return loaded_df, loaded_matrix, loaded_vec, loaded_indices


# Initialize in-memory model instances on server startup
df, tfidf_matrix, tfidf_vectorizer, indices = load_model_artifacts()


# ==============================================================================
# HELPER: DIVERSIFICATION FILTERING (LEVENSHTEIN DISTANCE)
# ==============================================================================
def calc_edit_distance_ratio(title1: str, title2: str) -> float:
    """Calculate normalized Levenshtein edit distance ratio between two titles.

    Used by the diversification filter to prune consecutive sequels and duplicate
    editions (e.g., 'Elden Ring' vs 'Elden Ring Deluxe Edition'). A ratio below 0.3
    indicates greater than 70% string similarity.

    Args:
        title1 (str): First game title string.
        title2 (str): Second game title string.

    Returns:
        float: Normalized edit distance between 0.0 (identical) and 1.0 (completely distinct).
    """
    t1 = str(title1).lower().strip()
    t2 = str(title2).lower().strip()
    max_len = max(len(t1), len(t2))
    if max_len == 0:
        return 0.0
    dist = Levenshtein.distance(t1, t2)
    return dist / max_len


def fuzzy_find_closest_titles(
    query: str,
    df_target: Optional[pd.DataFrame],
    limit: int = 7,
    max_distance_ratio: float = 0.35
) -> List[str]:
    """Find closest game titles using Levenshtein distance and token-level matching.

    Acts as a typo-tolerant fallback when exact or substring matching yields no results.

    Args:
        query (str): Cleaned search term.
        df_target (Optional[pd.DataFrame]): Game dataset DataFrame containing the 'name' column.
        limit (int, optional): Maximum number of suggested titles. Defaults to 7.
        max_distance_ratio (float, optional): Tolerable edit ratio threshold. Defaults to 0.35.

    Returns:
        List[str]: List of closest matching game titles ordered by similarity.
    """
    if df_target is None or not query:
        return []

    query_lower = query.lower().strip()
    q_len = len(query_lower)
    if q_len == 0:
        return []

    q_words = [w for w in re.findall(r'\w+', query_lower) if len(w) >= 2]

    # Dynamically adjust distance tolerance based on query character length
    if q_len <= 3:
        max_distance = 1
    elif q_len <= 6:
        max_distance = 2
    else:
        max_distance = max(2, int(q_len * max_distance_ratio))

    candidates: List[Tuple[str, int]] = []
    for name in df_target['name'].dropna().unique():
        name_lower = name.lower()
        n_len = len(name_lower)

        # Early exit heuristic: Skip comparison if character length difference is massive
        if abs(n_len - q_len) > max(max_distance + 3, int(q_len * 0.5)):
            continue

        # 1. Direct Levenshtein Distance
        dist = Levenshtein.distance(query_lower, name_lower)
        if dist <= max_distance:
            candidates.append((name, dist))
            continue

        # 2. Token-level matching for multi-word queries (e.g., 'elden rign' -> 'elden ring')
        if q_words and len(q_words) > 1:
            n_words = re.findall(r'\w+', name_lower)
            if len(n_words) == len(q_words):
                word_dists = [Levenshtein.distance(qw, nw) for qw, nw in zip(q_words, n_words)]
                total_word_dist = sum(word_dists)
                max_allowed_word_dist = sum(1 if len(qw) <= 4 else 2 for qw in q_words)
                if total_word_dist <= max_allowed_word_dist:
                    candidates.append((name, total_word_dist))

    # Rank candidates by smallest distance and minimum string length difference
    candidates.sort(key=lambda x: (x[1], abs(len(x[0]) - q_len)))

    # Deduplicate titles while preserving score ordering
    seen: set = set()
    result: List[str] = []
    for name, _ in candidates:
        if name not in seen:
            seen.add(name)
            result.append(name)
            if len(result) >= limit:
                break
    return result


def generate_dynamic_xai_text(
    genre_contributions: List[Tuple[str, float]],
    tag_contributions: List[Tuple[str, float]]
) -> str:
    """Generate dynamic textual Explainable AI (XAI) rationale based on top TF-IDF weights.

    Args:
        genre_contributions (List[Tuple[str, float]]): List of tuples containing (genre_name, score).
        tag_contributions (List[Tuple[str, float]]): List of tuples containing (tag_name, score).

    Returns:
        str: Human-readable Indonesian explanation detailing shared attributes.
    """
    candidates: List[Dict[str, Any]] = []
    for g, score in genre_contributions:
        candidates.append({'type': 'Genre', 'name': g, 'score': score})
    for t, score in tag_contributions:
        candidates.append({'type': 'Tag', 'name': t, 'score': score})

    # Sort descending by TF-IDF contribution score
    candidates.sort(key=lambda x: x['score'], reverse=True)

    if not candidates:
        return "Game ini direkomendasikan berdasarkan tingkat kemiripan fitur utama dengan game yang Anda pilih."

    if len(candidates) == 1:
        f1 = candidates[0]
        return f"Game ini direkomendasikan karena memiliki {f1['type']} {f1['name']} yang serupa dengan game yang Anda pilih."

    f1, f2 = candidates[0], candidates[1]
    return (
        f"Game ini direkomendasikan karena memiliki {f1['type']} {f1['name']} "
        f"dan {f2['type']} {f2['name']} yang serupa dengan game yang Anda pilih."
    )


def extract_tfidf_xai_explanation(
    target_pos: int,
    cand_pos: int,
    target_row: pd.Series,
    cand_row: pd.Series
) -> Dict[str, Any]:
    """Extract granular feature-level XAI rationale via TF-IDF dot-product element multiplication.

    Decomposes the Cosine Similarity score into feature-specific weight contributions
    across shared genres and tags, returning structured weights and narrative templates.

    Args:
        target_pos (int): Positional integer index of the query game in the TF-IDF matrix.
        cand_pos (int): Positional integer index of the recommended game in the TF-IDF matrix.
        target_row (pd.Series): DataFrame row of the query game.
        cand_row (pd.Series): DataFrame row of the recommended candidate game.

    Returns:
        Dict[str, Any]: Dictionary containing top matching features, scores, percentages,
            and bilingual narrative explanations (ID & EN).
    """
    if tfidf_matrix is None or tfidf_vectorizer is None:
        return {
            "top_matching_genres": [],
            "top_matching_tags": [],
            "top_features": {"genres": [], "tags": []},
            "other_features": {"genres": [], "tags": []},
            "xai_text_id": "Game ini direkomendasikan berdasarkan tingkat kemiripan fitur utama.",
            "xai_text_en": "This game is recommended based on overall key feature similarity.",
            "dynamic_text": "Game ini direkomendasikan berdasarkan tingkat kemiripan fitur utama."
        }

    # Extract sparse row vectors
    query_vec = tfidf_matrix[target_pos]
    cand_vec = tfidf_matrix[cand_pos]

    # Element-wise multiplication to isolate shared term weights
    prod = query_vec.multiply(cand_vec)
    vocab = tfidf_vectorizer.vocabulary_

    q_genres = [g.strip() for g in str(target_row.get('genres', '')).split(';') if g.strip()]
    c_genres = [g.strip() for g in str(cand_row.get('genres', '')).split(';') if g.strip()]
    common_genres = [g for g in c_genres if g in q_genres]

    q_tags = [t.strip() for t in str(target_row.get('tags', '')).split(';') if t.strip()]
    c_tags = [t.strip() for t in str(cand_row.get('tags', '')).split(';') if t.strip()]
    common_tags = [t for t in c_tags if t in q_tags]

    all_contributions: List[Dict[str, Any]] = []
    for g in common_genres:
        tokens = re.findall(r'\w+', g.lower())
        score = sum(prod[0, vocab[t]] for t in tokens if t in vocab)
        all_contributions.append({'category': 'genres', 'name': g, 'score': score})

    for t_item in common_tags:
        tokens = re.findall(r'\w+', t_item.lower())
        score = sum(prod[0, vocab[t]] for t in tokens if t in vocab)
        all_contributions.append({'category': 'tags', 'name': t_item, 'score': score})

    all_contributions.sort(key=lambda x: x['score'], reverse=True)

    # Division-by-zero guard for relative contribution percentage calculation
    total_score = sum(item['score'] for item in all_contributions)
    if total_score > 0:
        for item in all_contributions:
            item['pct'] = round((item['score'] / total_score) * 100)
    else:
        for item in all_contributions:
            item['pct'] = 0

    top_items = all_contributions[:3]
    other_items = all_contributions[3:]

    top_features: Dict[str, List[Dict[str, Any]]] = {"genres": [], "tags": []}
    for item in top_items:
        top_features[item['category']].append({
            "name": item['name'],
            "score": round(float(item['score']), 4),
            "pct": item['pct']
        })

    other_features: Dict[str, List[Dict[str, Any]]] = {"genres": [], "tags": []}
    for item in other_items:
        other_features[item['category']].append({
            "name": item['name'],
            "score": round(float(item['score']), 4),
            "pct": item['pct']
        })

    cand_name = str(cand_row.get('name', ''))
    # Deterministic narrative variant selection using CRC32 hash modulo
    variant_idx = zlib.crc32(cand_name.encode('utf-8')) % 4

    top_genres_str = ", ".join([item['name'] for item in top_items if item['category'] == 'genres'])
    top_tags_str = ", ".join([item['name'] for item in top_items if item['category'] == 'tags'])

    if top_genres_str and top_tags_str:
        templates_id = [
            f"Game ini direkomendasikan karena memiliki Genre {top_genres_str} serta Tag {top_tags_str} yang sama dengan game yang Anda pilih.",
            f"Kemiripan pada Genre {top_genres_str} serta Tag {top_tags_str} menjadi alasan utama game ini muncul sebagai rekomendasi.",
            f"Genre {top_genres_str} dan Tag {top_tags_str} pada game ini sejalan dengan preferensi dari game yang Anda cari.",
            f"Kami melihat kecocokan kuat di sisi Genre {top_genres_str} dan Tag {top_tags_str}, sehingga game ini masuk daftar rekomendasi Anda."
        ]
        templates_en = [
            f"This game is recommended because it shares the {top_genres_str} Genre and {top_tags_str} Tags with your selected game.",
            f"The similarity in {top_genres_str} Genre and {top_tags_str} Tags is the main reason this game appears as a recommendation.",
            f"The {top_genres_str} Genre and {top_tags_str} Tags in this title align directly with your search preference.",
            f"We found strong matching elements in {top_genres_str} Genre and {top_tags_str} Tags, placing this game on your recommendation list."
        ]
    elif top_genres_str:
        templates_id = [
            f"Game ini direkomendasikan karena memiliki kesamaan Genre {top_genres_str} dengan game pilihan Anda.",
            f"Kesamaan pada Genre {top_genres_str} menjadi faktor utama rekomendasi game ini.",
            f"Unsur Genre {top_genres_str} pada game ini sangat mirip dengan karakteristik game yang Anda cari.",
            f"Sistem menemukan kecocokan genre yang kuat pada {top_genres_str} dibanding game utama."
        ]
        templates_en = [
            f"This game is recommended because it shares the {top_genres_str} Genre with your selected game.",
            f"Shared characteristics in the {top_genres_str} Genre are the key factor behind this recommendation.",
            f"The {top_genres_str} Genre elements in this title closely match your selected game.",
            f"Our system identified a strong genre alignment around {top_genres_str}."
        ]
    elif top_tags_str:
        templates_id = [
            f"Game ini direkomendasikan karena memiliki kesamaan Tag {top_tags_str} dengan game pilihan Anda.",
            f"Kesamaan pada Tag {top_tags_str} menjadi penentu utama rekomendasi game ini.",
            f"Pengelompokan Tag {top_tags_str} pada game ini sangat sejalan dengan game pilihan Anda.",
            f"Sistem mendeteksi keterikatan tema yang kuat pada Tag {top_tags_str}."
        ]
        templates_en = [
            f"This game is recommended because it shares the {top_tags_str} Tags with your selected game.",
            f"Key matching elements in {top_tags_str} Tags are the main reason for this recommendation.",
            f"The {top_tags_str} Tags of this title align strongly with your reference game.",
            f"Our system detected a solid thematic match around the {top_tags_str} Tags."
        ]
    else:
        templates_id = [
            "Game ini direkomendasikan berdasarkan tingkat kemiripan fitur utama.",
            "Kemiripan karakteristik umum menjadi dasar rekomendasi game ini.",
            "Sistem mencocokkan profil keseluruhan game ini dengan preferensi pencarian Anda.",
            "Rekomendasi ini didasarkan pada kesamaan atribut cerita dan mekanisme permainan."
        ]
        templates_en = [
            "This game is recommended based on overall key feature similarity.",
            "Overall characteristic similarity serves as the foundation for this recommendation.",
            "The system matched the general profile of this game with your search preference.",
            "This recommendation is derived from shared narrative and gameplay mechanics."
        ]

    xai_text_id = templates_id[variant_idx]
    xai_text_en = templates_en[variant_idx]

    return {
        "top_matching_genres": [g['name'] for g in all_contributions if g['category'] == 'genres'][:3],
        "top_matching_tags": [t['name'] for t in all_contributions if t['category'] == 'tags'][:5],
        "top_features": top_features,
        "other_features": other_features,
        "xai_text_id": xai_text_id,
        "xai_text_en": xai_text_en,
        "dynamic_text": xai_text_id
    }


def get_corpus_word_count(row: pd.Series) -> int:
    """Calculate total token count within combined game corpus fields.

    Args:
        row (pd.Series): Single game record from metadata DataFrame.

    Returns:
        int: Number of discrete words across descriptions, genres, and tags.
    """
    desc = str(row.get('short_description', '')).strip()
    if not desc or desc == 'nan':
        desc = str(row.get('detailed_description', '')).strip()
    if desc == 'nan':
        desc = ""
    genres = str(row.get('genres', '')).replace(';', ' ') if str(row.get('genres', '')).strip() != 'nan' else ""
    tags = str(row.get('tags', '')).replace(';', ' ') if str(row.get('tags', '')).strip() != 'nan' else ""
    text = f"{desc} {genres} {tags}".strip()
    return len([w for w in text.split() if w])


def check_is_sparse_corpus(row: pd.Series) -> bool:
    """Verify if a game entry suffers from sparse or insufficient metadata.

    Args:
        row (pd.Series): Single game metadata row.

    Returns:
        bool: True if descriptions are missing or total token count is under 15.
    """
    empty_s = str(row.get('short_description', '')).strip() in ['', 'nan']
    empty_d = str(row.get('detailed_description', '')).strip() in ['', 'nan']
    both_empty = empty_s and empty_d
    word_cnt = get_corpus_word_count(row)
    return bool(both_empty or word_cnt < 15)


# ==============================================================================
# ALGORITMA REKOMENDASI UTAMA DENGAN IN-MEMORY LRU CACHE
# ==============================================================================
@lru_cache(maxsize=256)
def _cached_get_recommendations(
    query_clean: str,
    top_n: int = 12
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Internal cached recommendation engine with tie-breaking and diversification.

    Args:
        query_clean (str): Normalized lowercase title query.
        top_n (int, optional): Number of recommendations to retrieve. Defaults to 12.

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[str]]:
            Tuple containing recommendation result payload (or None on failure)
            and error message string (or None on success).
    """
    t0 = time.time()

    # Match exact title first
    matches = df[df['name'].str.lower() == query_clean]
    if matches.empty:
        # Fallback to substring matching
        matches = df[df['name'].str.lower().str.contains(query_clean, regex=False, na=False)]

    corrected_from = None
    fuzzy_suggestions: List[str] = []

    if matches.empty:
        # Typo-tolerant fallback: Search via Levenshtein fuzzy distance
        closest_candidates = fuzzy_find_closest_titles(query_clean, df, limit=5)
        if closest_candidates:
            best_match = closest_candidates[0]
            dist = Levenshtein.distance(query_clean, best_match.lower())

            # Auto-correct if distance is tight (obvious typographical error)
            if dist <= (2 if len(query_clean) >= 5 else 1):
                corrected_from = query_clean
                matches = df[df['name'].str.lower() == best_match.lower()]
                if matches.empty:
                    matches = df[df['name'].str.lower().str.contains(best_match.lower(), regex=False, na=False)]
            else:
                fuzzy_suggestions = closest_candidates

    if matches.empty:
        return {
            'error_type': 'not_found',
            'fuzzy_suggestions': fuzzy_suggestions
        }, f"Game '{query_clean}' tidak ditemukan dalam sistem kami."

    # CRITICAL: Convert DataFrame index label to integer positional row index
    # Required for iloc slicing and accessing rows in scipy sparse CSR matrix
    target_pos = df.index.get_loc(matches.index[0])
    target_row = df.iloc[target_pos]
    game_target_name = target_row['name']
    t_match = time.time()

    # 1. REAL-TIME COSINE SIMILARITY CALCULATION
    query_vec = tfidf_matrix[target_pos]
    sim_scores = linear_kernel(query_vec, tfidf_matrix).flatten()
    t_cosine = time.time()

    # 2. TIE-BREAKER & SOFT PENALTY (PRECOMPUTED NUMPY ARRAYS FOR HIGH-SPEED VECTORIZED ACCESS)
    is_sparse_arr = df['is_sparse_corpus'].values if 'is_sparse_corpus' in df.columns else np.zeros(len(df), dtype=bool)
    pos_rev_arr = df['positive_reviews'].values if 'positive_reviews' in df.columns else np.zeros(len(df), dtype=float)

    t_sparse_start = time.time()
    # Apply soft penalty (0.90x) to sparse items to prevent spurious high similarities
    penalty_factors = np.where(is_sparse_arr, 0.90, 1.00)
    final_ranking_scores = sim_scores * penalty_factors
    t_sparse_sum = time.time() - t_sparse_start

    # Exclude the target game itself by nullifying its score
    final_ranking_scores[target_pos] = -1.0

    # Fast multi-key sorting using numpy lexsort: secondary (pos_rev_arr), primary (final_ranking_scores)
    t_sort_start = time.time()
    sorted_indices = np.lexsort((pos_rev_arr, final_ranking_scores))[::-1]
    t_sort = time.time()

    candidates: List[Dict[str, Any]] = []
    # Evaluate top 100 candidates through diversification pipeline to optimize throughput
    for idx in sorted_indices[:100]:
        if final_ranking_scores[idx] < 0:
            break
        candidates.append({
            'index': int(idx),
            'sim_score': float(sim_scores[idx]),
            'final_score': float(final_ranking_scores[idx]),
            'is_sparse_corpus': bool(is_sparse_arr[idx]),
            'positive_reviews': float(pos_rev_arr[idx])
        })

    # 3. DIVERSIFICATION FILTERING & XAI EXTRACTION
    accepted_recommendations: List[Dict[str, Any]] = []
    accepted_titles: List[str] = [game_target_name]

    t_xai_sum = 0.0
    t_lev_sum = 0.0

    for cand in candidates:
        cand_row = df.iloc[cand['index']]
        cand_name = cand_row['name']

        t_lev_start = time.time()
        is_duplicate_sequel = False
        # Filter out sequels/duplicate editions using normalized edit distance ratio
        for acc_title in accepted_titles:
            edit_ratio = calc_edit_distance_ratio(cand_name, acc_title)
            if edit_ratio < 0.3:
                is_duplicate_sequel = True
                break
        t_lev_sum += (time.time() - t_lev_start)

        if not is_duplicate_sequel:
            accepted_titles.append(cand_name)

            sim_percentage = round(cand['sim_score'] * 100, 1)
            sim_score_val = round(float(cand['sim_score']), 2)

            t_xai_start = time.time()
            xai_explanation = extract_tfidf_xai_explanation(
                target_pos, cand['index'], target_row, cand_row
            )
            t_xai_sum += (time.time() - t_xai_start)

            cand_pos = float(cand_row.get('positive_reviews', 0))
            cand_tot = float(cand_row.get('total_reviews', 0))
            cand_rating_score = float(cand_row.get('rating_score', 0))
            # Calculate review ratio fallback if score is zero but total reviews exist
            if cand_rating_score == 0 and cand_tot > 0:
                cand_rating_score = round((cand_pos / cand_tot) * 100, 1)

            rec_item = {
                'game_title': str(cand_name),
                'steam_appid': int(cand_row['steam_appid']),
                'name': str(cand_name),
                'price': float(cand_row['price']),
                'genres': str(cand_row['genres']),
                'tags': str(cand_row['tags']),
                'header_image': str(cand_row['header_image']),
                'short_description': str(cand_row['short_description']),
                'detailed_description': str(cand_row['detailed_description']),
                'rating_score': cand_rating_score,
                'rating': str(cand_row.get('rating', 'Very Positive')),
                'positive_reviews': cand_pos,
                'total_reviews': cand_tot,
                'similarity_score': sim_score_val,
                'similarity_percentage': sim_percentage,
                'similarity_pct': int(round(sim_percentage)),
                'cosine_score': f"{sim_score_val:.2f}",
                'is_sparse_corpus': cand['is_sparse_corpus'],
                'top_features': xai_explanation['top_features'],
                'other_features': xai_explanation['other_features'],
                'xai_text_id': xai_explanation['xai_text_id'],
                'xai_text_en': xai_explanation['xai_text_en'],
                'xai_explanation': xai_explanation,
                'explanation_en': xai_explanation['xai_text_en']
            }
            accepted_recommendations.append(rec_item)

        if len(accepted_recommendations) >= top_n:
            break

    # Flag tight clusters (recommendations with nearly identical top scores)
    if accepted_recommendations:
        max_sim_pct = max(r['similarity_percentage'] for r in accepted_recommendations)
        tight_count = sum(1 for r in accepted_recommendations if (max_sim_pct - r['similarity_percentage']) <= 1.5)
        is_tight_group = tight_count >= 3

        for r in accepted_recommendations:
            r['is_tight_cluster'] = bool(is_tight_group and (max_sim_pct - r['similarity_percentage']) <= 1.5)

    target_pos_val = float(target_row.get('positive_reviews', 0))
    target_tot_val = float(target_row.get('total_reviews', 0))
    target_rating_score = float(target_row.get('rating_score', 0))
    if target_rating_score == 0 and target_tot_val > 0:
        target_rating_score = round((target_pos_val / target_tot_val) * 100, 1)

    target_data = {
        'steam_appid': int(target_row['steam_appid']),
        'name': str(target_row['name']),
        'price': float(target_row['price']),
        'genres': str(target_row['genres']),
        'tags': str(target_row['tags']),
        'header_image': str(target_row['header_image']),
        'short_description': str(target_row['short_description']),
        'detailed_description': str(target_row['detailed_description']),
        'rating_score': target_rating_score,
        'rating': str(target_row.get('rating', 'Very Positive')),
        'positive_reviews': target_pos_val,
        'total_reviews': target_tot_val,
        'similarity_score': 100.0,
        'is_sparse_corpus': check_is_sparse_corpus(target_row),
        'explanation': "Game target pencarian utama."
    }
    t_end = time.time()

    # Latency profiling log
    try:
        print(f"\n=== PROFILING RESULTS FOR '{query_clean}' (top_n={top_n}) ===")
        print(f"1. Target Search Match : {(t_match - t0)*1000:.2f} ms")
        print(f"2. Cosine Similarity   : {(t_cosine - t_match)*1000:.2f} ms")
        print(f"3. Soft Penalty Vector : {t_sparse_sum*1000:.2f} ms")
        print(f"4. Sorting Candidates  : {(t_sort - t_sort_start)*1000:.2f} ms")
        print(f"5. Levenshtein Loop    : {t_lev_sum*1000:.2f} ms")
        print(f"6. XAI Extraction      : {t_xai_sum*1000:.2f} ms")
        print(f"TOTAL EXCLUSIVELY      : {(t_end - t0)*1000:.2f} ms\n")
    except Exception:
        pass

    return {
        'target_game': target_data,
        'recommendations': accepted_recommendations,
        'corrected_from': corrected_from
    }, None


def get_recommendations_data(
    title: str,
    top_n: int = 12
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Public wrapper to fetch recommendations through the in-memory LRU cache.

    Args:
        title (str): Raw or sanitized query game title.
        top_n (int, optional): Recommendation count threshold. Defaults to 12.

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[str]]: Result payload or error message.
    """
    if df is None or tfidf_matrix is None:
        return None, "Model belum dimuat ke RAM peladen."
    query_clean = sanitize_input(title).lower().strip()
    return _cached_get_recommendations(query_clean, top_n)


# ==============================================================================
# ROUTING API & WEB
# ==============================================================================

@app.route('/', methods=['GET', 'POST'])
@limiter.limit("30 per minute")
def home() -> str:
    """Handle home page rendering and recommendation queries via GET or POST.

    Returns:
        str: Rendered HTML template index.html.
    """
    search_query = ""
    error = None
    target_game = None
    recommendations = None

    top_n = request.form.get('top_n', type=int) or request.args.get('top_n', type=int) or 12

    if request.method == 'POST':
        search_query = sanitize_input(request.form.get('game_title', ''))
    else:
        search_query = sanitize_input(request.args.get('q', '') or request.args.get('game_title', ''))

    popular_suggestions = ["Elden Ring", "Cyberpunk 2077", "The Witcher 3", "Palworld", "Baldur's Gate 3", "Grand Theft Auto V"]
    suggestions = None
    corrected_from = None

    if search_query:
        result, error_msg = get_recommendations_data(search_query, top_n=top_n)
        if error_msg:
            error = error_msg
            if isinstance(result, dict) and result.get('fuzzy_suggestions'):
                suggestions = result['fuzzy_suggestions']
            else:
                suggestions = popular_suggestions
        else:
            target_game = result['target_game']
            rec_list = result['recommendations']
            corrected_from = result.get('corrected_from')

            # Prepend target game as index 0 row for Jinja template rendering compatibility
            combined_list = [target_game] + rec_list
            recommendations = pd.DataFrame(combined_list)

    return render_template(
        'index.html',
        search_query=search_query,
        top_n=top_n,
        actual_title=target_game['name'] if target_game else search_query,
        target_game=target_game,
        recommendations=recommendations if target_game else None,
        suggestions=suggestions,
        corrected_from=corrected_from,
        error=error
    )


@app.route('/robots.txt', methods=['GET'])
def robots_txt() -> Tuple[str, int, Dict[str, str]]:
    """Serve robots.txt file allowing crawler indexing of root and sitemap.

    Returns:
        Tuple[str, int, Dict[str, str]]: Plain text response, HTTP status, and headers.
    """
    content = "User-agent: *\nAllow: /\nSitemap: " + request.url_root.rstrip('/') + "/sitemap.xml\n"
    return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}


@app.route('/sitemap.xml', methods=['GET'])
def sitemap_xml() -> Tuple[str, int, Dict[str, str]]:
    """Serve XML sitemap for search engine optimization and discovery.

    Returns:
        Tuple[str, int, Dict[str, str]]: XML payload, HTTP status code, and Content-Type headers.
    """
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>{request.url_root.rstrip('/')}/</loc>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>
</urlset>"""
    return xml_content, 200, {'Content-Type': 'application/xml; charset=utf-8'}


@app.route('/api/search-suggestions', methods=['GET'])
@limiter.limit("60 per minute")
def api_search_titles() -> Response:
    """Provide title suggestions for frontend search autocomplete.

    Accepts 'term' or 'q' parameter and performs substring search followed by fuzzy fallback.

    Returns:
        Response: JSON array containing up to 7 matching game titles.
    """
    raw_query = request.args.get('term', '').strip().lower() or request.args.get('q', '').strip().lower()
    query = sanitize_input(raw_query).lower()
    if not query or df is None:
        return jsonify([])

    matches = df[df['name'].fillna('').str.lower().str.contains(query, regex=False, na=False)]['name'].head(7).tolist()
    if not matches:
        matches = fuzzy_find_closest_titles(query, df, limit=7)

    return jsonify(matches)


@app.route('/api/recommend', methods=['GET', 'POST'])
@limiter.limit("30 per minute")
def api_recommend() -> Tuple[Response, int]:
    """REST API endpoint returning structured game recommendations and XAI metrics in JSON.

    Supports query parameter 'q' or 'game_title' via GET, as well as JSON body via POST.

    Returns:
        Tuple[Response, int]: JSON response payload accompanied by appropriate HTTP status code.
    """
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        game_title = data.get('game_title') or request.form.get('game_title', '')
    else:
        game_title = request.args.get('q') or request.args.get('game_title', '')

    game_title = sanitize_input(game_title)
    if not game_title:
        return jsonify({'status': 'error', 'message': 'Parameter game_title tidak boleh kosong.'}), 400

    result, error_msg = get_recommendations_data(game_title)
    if error_msg:
        status_code = 404 if 'tidak ditemukan' in error_msg else 500
        return jsonify({'status': 'error', 'message': error_msg}), status_code

    return jsonify({
        'status': 'success',
        'data': result
    }), 200


# ==============================================================================
# ERROR HANDLER UNTUK KEAMANAN STACK TRACE
# ==============================================================================
@app.errorhandler(429)
def ratelimit_handler(e: Any) -> Tuple[Union[Response, str], int]:
    """Handle rate limit breach without leaking server internals.

    Args:
        e (Any): Raised exception or error details.

    Returns:
        Tuple[Union[Response, str], int]: JSON response for API or rendered HTML for web.
    """
    if request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'message': 'Batas permintaan terlampaui. Silakan tunggu 1 menit.'}), 429
    return render_template('index.html', error="Batas permintaan pencarian terlampaui. Silakan tunggu 1 menit."), 429


@app.errorhandler(500)
def internal_error_handler(e: Any) -> Tuple[Union[Response, str], int]:
    """Handle internal application errors securely.

    Args:
        e (Any): Raised internal error.

    Returns:
        Tuple[Union[Response, str], int]: Sanitized generic error payload.
    """
    if request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'message': 'Terjadi kesalahan internal peladen.'}), 500
    return render_template('index.html', error="Terjadi kesalahan internal peladen saat mengolah rekomendasi."), 500


@app.errorhandler(404)
def not_found_handler(e: Any) -> Tuple[Union[Response, str], int]:
    """Handle resource not found errors.

    Args:
        e (Any): 404 exception object.

    Returns:
        Tuple[Union[Response, str], int]: Not found response for API or web view.
    """
    if request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'message': 'Endpoint tidak ditemukan.'}), 404
    return render_template('index.html', error="Halaman tidak ditemukan."), 404


if __name__ == '__main__':
    # Production note: debug disabled to prevent stack trace leaks and arbitrary execution
    app.run(debug=False, port=5000)
