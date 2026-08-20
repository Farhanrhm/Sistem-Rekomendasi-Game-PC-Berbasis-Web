from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
import pickle
import Levenshtein
import os

app = Flask(__name__)

# ==============================================================================
# 2. BATASAN PEMODELAN & MEMORI & 3. LAZY LOADING
# Muat file .pkl ke dalam RAM peladen HANYA saat aplikasi Flask menyala (bukan di dalam fungsi route)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("[LAZY LOADING] Memuat file model .pkl ke dalam RAM peladen...")
df = None
tfidf_matrix = None
tfidf_vectorizer = None
indices = None

try:
    with open(os.path.join(BASE_DIR, 'models', 'game_data.pkl'), 'rb') as f:
        df = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'models', 'tfidf_matrix.pkl'), 'rb') as f:
        tfidf_matrix = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl'), 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'models', 'indices.pkl'), 'rb') as f:
        indices = pickle.load(f)
    print("[OK] Model dan data sparse TF-IDF berhasil dimuat ke RAM peladen!")

except Exception as e:
    print(f"[ERROR] Error memuat file model .pkl: {e}")
    print("Harap pastikan Anda sudah menjalankan 3_build_model.py terlebih dahulu.")


# ==============================================================================
# HELPER: DIVERSIFICATION FILTERING (LEVENSHTEIN DISTANCE)
# ==============================================================================
def calc_edit_distance_ratio(title1, title2):
    """
    BATASAN PROPOSAL: Diversification Filtering
    Menghitung rasio Levenshtein Distance antara dua judul game.
    Rasio < 0.3 berarti kemiripan string > 70% (mengeliminasi sekuel game yang berulang).
    """
    t1 = str(title1).lower().strip()
    t2 = str(title2).lower().strip()
    max_len = max(len(t1), len(t2))
    if max_len == 0:
        return 0.0
    dist = Levenshtein.distance(t1, t2)
    return dist / max_len


def extract_dominant_genre(target_genres_str, rec_genres_str):
    """
    Ekstrak genre dominan yang beririsan antara game target dan game rekomendasi.
    """
    if not isinstance(target_genres_str, str) or not isinstance(rec_genres_str, str):
        return "Genre"
    target_genres = [g.strip() for g in target_genres_str.split(';') if g.strip()]
    rec_genres = [g.strip() for g in rec_genres_str.split(';') if g.strip()]
    
    # Cari genre yang sama
    common = [g for g in rec_genres if g in target_genres]
    if common:
        return common[0]
    elif rec_genres:
        return rec_genres[0]
    return "Genre"


from functools import lru_cache

# ==============================================================================
# ALGORITMA REKOMENDASI UTAMA DENGAN IN-MEMORY LRU CACHE
# ==============================================================================
@lru_cache(maxsize=256)
def _cached_get_recommendations(query_clean, top_n=10):
    """
    IN-MEMORY LRU CACHE (OPTIMASI LATENSI ENGINERING):
    Menyimpan hasil kalkulasi Cosine Similarity & Diversification untuk pencarian
    yang pernah dilakukan. Memangkas latensi ulang dari ~50ms menjadi 0ms (instan).
    """
    matches = df[df['name'].str.lower() == query_clean]
    if matches.empty:
        matches = df[df['name'].str.lower().str.contains(query_clean, regex=False, na=False)]
        
    if matches.empty:
        return None, f"Game '{query_clean}' tidak ditemukan dalam database."

    target_idx = matches.index[0]
    target_row = df.iloc[target_idx]
    game_target_name = target_row['name']

    # 1. REAL-TIME COSINE SIMILARITY
    query_vec = tfidf_matrix[target_idx]
    sim_scores = linear_kernel(query_vec, tfidf_matrix).flatten()

    # 2. TIE-BREAKER MECHANISM
    candidates = []
    for i, score in enumerate(sim_scores):
        if i == target_idx:
            continue  # Abaikan game itu sendiri
        pos_rev = float(df.iloc[i].get('positive_reviews', 0))
        candidates.append({
            'index': i,
            'sim_score': float(score),
            'positive_reviews': pos_rev
        })

    # Urutkan primer berdasarkan sim_score (descending), sekunder berdasarkan positive_reviews (descending)
    candidates = sorted(
        candidates,
        key=lambda x: (x['sim_score'], x['positive_reviews']),
        reverse=True
    )

    # 3. DIVERSIFICATION FILTERING (LEVENSHTEIN DISTANCE)
    accepted_recommendations = []
    accepted_titles = [game_target_name]

    for cand in candidates:
        cand_row = df.iloc[cand['index']]
        cand_name = cand_row['name']
        
        is_duplicate_sequel = False
        for acc_title in accepted_titles:
            edit_ratio = calc_edit_distance_ratio(cand_name, acc_title)
            if edit_ratio < 0.3:
                is_duplicate_sequel = True
                break

        if not is_duplicate_sequel:
            accepted_titles.append(cand_name)
            
            sim_percentage = round(cand['sim_score'] * 100, 1)
            dominant_genre = extract_dominant_genre(target_row['genres'], cand_row['genres'])
            
            # 4. EXPLAINABLE AI (XAI) TEMPLATE-BASED GENERATION
            xai_explanation = (
                f"Game {cand_name} direkomendasikan karena memiliki kesamaan {dominant_genre} "
                f"dengan skor {sim_percentage}% terhadap {game_target_name}."
            )

            cand_pos = float(cand_row.get('positive_reviews', 0))
            cand_tot = float(cand_row.get('total_reviews', 0))
            cand_rating_score = float(cand_row.get('rating_score', 0))
            if cand_rating_score == 0 and cand_tot > 0:
                cand_rating_score = round((cand_pos / cand_tot) * 100, 1)

            rec_item = {
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
                'similarity_score': sim_percentage,
                'explanation': xai_explanation
            }
            accepted_recommendations.append(rec_item)

        if len(accepted_recommendations) >= top_n:
            break

    target_pos = float(target_row.get('positive_reviews', 0))
    target_tot = float(target_row.get('total_reviews', 0))
    target_rating_score = float(target_row.get('rating_score', 0))
    if target_rating_score == 0 and target_tot > 0:
        target_rating_score = round((target_pos / target_tot) * 100, 1)

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
        'positive_reviews': target_pos,
        'total_reviews': target_tot,
        'similarity_score': 100.0,
        'explanation': f"Game target pencarian utama."
    }

    return {
        'target_game': target_data,
        'recommendations': accepted_recommendations
    }, None


def get_recommendations_data(title, top_n=10):
    """
    Wrapper publik yang memanggil cache LRU In-Memory.
    """
    if df is None or tfidf_matrix is None:
        return None, "Model belum dimuat ke RAM peladen."
    query_clean = title.lower().strip()
    return _cached_get_recommendations(query_clean, top_n)


# ==============================================================================
# ROUTING API & WEB
# ==============================================================================

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Halaman Web Utama - Mendukung pencarian via GET (?q=...) dan POST (form submission)
    """
    search_query = ""
    error = None
    target_game = None
    recommendations = []

    if request.method == 'POST':
        search_query = request.form.get('game_title', '').strip()
    else:
        search_query = request.args.get('q', '').strip() or request.args.get('game_title', '').strip()

    if search_query:
        result, error_msg = get_recommendations_data(search_query)
        if error_msg:
            error = error_msg
        else:
            target_game = result['target_game']
            rec_list = result['recommendations']
            
            # Gabungkan target game di baris 0 agar kompatibel dengan template index.html
            combined_list = [target_game] + rec_list
            recommendations = pd.DataFrame(combined_list)

    return render_template(
        'index.html',
        search_query=search_query,
        actual_title=target_game['name'] if target_game else search_query,
        recommendations=recommendations if target_game else None,
        error=error
    )


@app.route('/api/search_autocomplete', methods=['GET'])
@app.route('/api/search_titles', methods=['GET'])
def api_search_titles():
    """
    Endpoint Autocomplete Judul Game untuk Frontend.
    """
    query = request.args.get('term', '').strip().lower() or request.args.get('q', '').strip().lower()
    if not query or df is None:
        return jsonify([])
    
    matches = df[df['name'].str.lower().str.contains(query, regex=False)]['name'].head(10).tolist()
    return jsonify(matches)


@app.route('/api/recommend', methods=['GET', 'POST'])
def api_recommend():
    """
    BATASAN PROPOSAL: Endpoint API JSON untuk Frontend HTML/JS.
    Mengembalikan response JSON lengkap dengan target game dan rekomendasi Top-N.
    """
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        game_title = data.get('game_title') or request.form.get('game_title', '')
    else:
        game_title = request.args.get('q') or request.args.get('game_title', '')

    game_title = game_title.strip()
    if not game_title:
        return jsonify({'status': 'error', 'message': 'Parameter game_title tidak boleh kosong.'}), 400

    result, error_msg = get_recommendations_data(game_title)
    if error_msg:
        return jsonify({'status': 'error', 'message': error_msg}), 404 if 'tidak ditemukan' in error_msg else 500

    return jsonify({
        'status': 'success',
        'data': result
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)


