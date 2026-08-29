from flask import Flask, render_template, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
import pickle
import Levenshtein
import os
import re
import html

app = Flask(__name__)

# ==============================================================================
# 1. PENGAMANAN RATE LIMITING & CORS
# Rate limiter menggunakan RAM (memory://) agar latensi tetap 0ms
# ==============================================================================
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["300 per day", "100 per hour"],
    storage_uri="memory://"
)

CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5000", "http://127.0.0.1:5000", "https://levelfind.vercel.app"]}})

def sanitize_input(user_input):
    """
    Sanitasi kata kunci pencarian dari potensi injeksi karakter berbahaya/HTML tags.
    """
    if not user_input or not isinstance(user_input, str):
        return ""
    clean_str = re.sub(r'<[^>]*>', '', user_input)
    clean_str = html.escape(clean_str.strip())
    return clean_str[:100]

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


def generate_dynamic_xai_text(genre_contributions, tag_contributions):
    """
    Fungsi pembantu untuk menghasilkan narasi XAI akademis secara dinamis 
    berdasarkan 2 fitur dengan kontribusi TF-IDF tertinggi.
    """
    candidates = []
    for g, score in genre_contributions:
        candidates.append({'type': 'Genre', 'name': g, 'score': score})
    for t, score in tag_contributions:
        candidates.append({'type': 'Tag', 'name': t, 'score': score})

    # Urutkan berdasarkan kontribusi skor TF-IDF descending
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


def extract_tfidf_xai_explanation(target_idx, cand_idx, target_row, cand_row):
    """
    Ekstraksi XAI berdasarkan bobot kontribusi TF-IDF riil (tfidf_query[term] * tfidf_candidate[term]).
    Mengidentifikasi genre dan tag yang beririsan, menghitung bobot kontribusinya,
    mengurutkan secara descending, dan mengambil top 3-5 term untuk JSON XAI data.
    """
    if tfidf_matrix is None or tfidf_vectorizer is None:
        return {
            "top_matching_genres": [],
            "top_matching_tags": [],
            "dynamic_text": "Game ini direkomendasikan berdasarkan tingkat kemiripan fitur utama."
        }

    query_vec = tfidf_matrix[target_idx]
    cand_vec = tfidf_matrix[cand_idx]
    
    # Hitung kontribusi perkalian elemen TF-IDF (element-wise product)
    prod = query_vec.multiply(cand_vec)
    vocab = tfidf_vectorizer.vocabulary_
    
    # Irisan Genres antara game target dan game kandidat
    q_genres = [g.strip() for g in str(target_row.get('genres', '')).split(';') if g.strip()]
    c_genres = [g.strip() for g in str(cand_row.get('genres', '')).split(';') if g.strip()]
    common_genres = [g for g in c_genres if g in q_genres]
    
    # Irisan Tags antara game target dan game kandidat
    q_tags = [t.strip() for t in str(target_row.get('tags', '')).split(';') if t.strip()]
    c_tags = [t.strip() for t in str(cand_row.get('tags', '')).split(';') if t.strip()]
    common_tags = [t for t in c_tags if t in q_tags]

    # Hitung kontribusi bobot TF-IDF untuk genre yang beririsan
    genre_contributions = []
    for g in common_genres:
        tokens = re.findall(r'\w+', g.lower())
        score = sum(prod[0, vocab[t]] for t in tokens if t in vocab)
        genre_contributions.append((g, score))
    genre_contributions.sort(key=lambda x: x[1], reverse=True)
    top_matching_genres = [g[0] for g in genre_contributions[:3]]

    # Hitung kontribusi bobot TF-IDF untuk tag yang beririsan
    tag_contributions = []
    for t_item in common_tags:
        tokens = re.findall(r'\w+', t_item.lower())
        score = sum(prod[0, vocab[t]] for t in tokens if t in vocab)
        tag_contributions.append((t_item, score))
    tag_contributions.sort(key=lambda x: x[1], reverse=True)
    top_matching_tags = [t[0] for t in tag_contributions[:5]]

    # Ambil top 2 fitur dengan kontribusi tertinggi untuk lokalisasi frontend
    candidates = []
    for g, score in genre_contributions:
        candidates.append({'type': 'Genre', 'name': g, 'score': score})
    for t, score in tag_contributions:
        candidates.append({'type': 'Tag', 'name': t, 'score': score})
    candidates.sort(key=lambda x: x['score'], reverse=True)
    top_features = [{'type': item['type'], 'name': item['name']} for item in candidates[:2]]

    # Hasilkan narasi XAI akademis dinamis dari 2 fitur kontribusi TF-IDF tertinggi
    dynamic_text = generate_dynamic_xai_text(genre_contributions, tag_contributions)

    return {
        "top_matching_genres": top_matching_genres,
        "top_matching_tags": top_matching_tags,
        "top_features": top_features,
        "dynamic_text": dynamic_text
    }


from functools import lru_cache

# ==============================================================================
# ALGORITMA REKOMENDASI UTAMA DENGAN IN-MEMORY LRU CACHE
# ==============================================================================
@lru_cache(maxsize=256)
def _cached_get_recommendations(query_clean, top_n=12):
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
            sim_score_val = round(float(cand['sim_score']), 2)
            
            # 4. EXPLAINABLE AI (XAI) TF-IDF FEATURE IMPORTANCE EXTRACTION
            xai_explanation = extract_tfidf_xai_explanation(
                target_idx, cand['index'], target_row, cand_row
            )

            cand_pos = float(cand_row.get('positive_reviews', 0))
            cand_tot = float(cand_row.get('total_reviews', 0))
            cand_rating_score = float(cand_row.get('rating_score', 0))
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
                'xai_explanation': xai_explanation,
                'explanation_en': xai_explanation['dynamic_text']
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


def get_recommendations_data(title, top_n=12):
    """
    Wrapper publik yang memanggil cache LRU In-Memory.
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
def home():
    """
    Halaman Web Utama - Mendukung pencarian via GET (?q=...) dan POST (form submission)
    """
    search_query = ""
    error = None
    target_game = None
    recommendations = []

    top_n = request.form.get('top_n', type=int) or request.args.get('top_n', type=int) or 12

    if request.method == 'POST':
        search_query = sanitize_input(request.form.get('game_title', ''))
    else:
        search_query = sanitize_input(request.args.get('q', '') or request.args.get('game_title', ''))

    if search_query:
        result, error_msg = get_recommendations_data(search_query, top_n=top_n)
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
        top_n=top_n,
        actual_title=target_game['name'] if target_game else search_query,
        recommendations=recommendations if target_game else None,
        error=error
    )


@app.route('/api/search-suggestions', methods=['GET'])
@app.route('/api/search_autocomplete', methods=['GET'])
@app.route('/api/search_titles', methods=['GET'])
@limiter.limit("60 per minute")
def api_search_titles():
    """
    Endpoint Autocomplete Judul Game untuk Frontend (Maksimal 7 judul game).
    """
    raw_query = request.args.get('term', '').strip().lower() or request.args.get('q', '').strip().lower()
    query = sanitize_input(raw_query).lower()
    if not query or df is None:
        return jsonify([])
    
    matches = df[df['name'].fillna('').str.lower().str.contains(query, regex=False, na=False)]['name'].head(7).tolist()
    return jsonify(matches)


@app.route('/api/recommend', methods=['GET', 'POST'])
@limiter.limit("30 per minute")
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

    game_title = sanitize_input(game_title)
    if not game_title:
        return jsonify({'status': 'error', 'message': 'Parameter game_title tidak boleh kosong.'}), 400

    result, error_msg = get_recommendations_data(game_title)
    if error_msg:
        return jsonify({'status': 'error', 'message': error_msg}), 404 if 'tidak ditemukan' in error_msg else 500

    return jsonify({
        'status': 'success',
        'data': result
    })


# ==============================================================================
# ERROR HANDLER UNTUK KEAMANAN STACK TRACE
# ==============================================================================
@app.errorhandler(429)
def ratelimit_handler(e):
    if request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'message': 'Batas permintaan terlampaui. Silakan tunggu 1 menit.'}), 429
    return render_template('index.html', error="Batas permintaan pencarian terlampaui. Silakan tunggu 1 menit."), 429

@app.errorhandler(500)
def internal_error_handler(e):
    if request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'message': 'Terjadi kesalahan internal peladen.'}), 500
    return render_template('index.html', error="Terjadi kesalahan internal peladen saat mengolah rekomendasi."), 500

@app.errorhandler(404)
def not_found_handler(e):
    if request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'message': 'Endpoint tidak ditemukan.'}), 404
    return render_template('index.html', error="Halaman tidak ditemukan."), 404


if __name__ == '__main__':
    # Modus produksi: Nonaktifkan debug mode untuk mencegah kebocoran stack trace
    app.run(debug=False, port=5000)


