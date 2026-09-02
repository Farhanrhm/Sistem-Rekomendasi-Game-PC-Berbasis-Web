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
import zlib

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
    Sanitasi kata kunci pencarian dari potensi injeksi HTML tags.
    """
    if not user_input or not isinstance(user_input, str):
        return ""
    clean_str = html.unescape(user_input)
    clean_str = re.sub(r'<[^>]*>', '', clean_str)
    return clean_str.strip()[:100]

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
    Ekstraksi XAI berdasarkan bobot kontribusi TF-IDF riil.
    Membagi fitur beririsan menjadi top_features (bobot tertinggi) & other_features.
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

    query_vec = tfidf_matrix[target_idx]
    cand_vec = tfidf_matrix[cand_idx]
    
    prod = query_vec.multiply(cand_vec)
    vocab = tfidf_vectorizer.vocabulary_
    
    q_genres = [g.strip() for g in str(target_row.get('genres', '')).split(';') if g.strip()]
    c_genres = [g.strip() for g in str(cand_row.get('genres', '')).split(';') if g.strip()]
    common_genres = [g for g in c_genres if g in q_genres]

    q_tags = [t.strip() for t in str(target_row.get('tags', '')).split(';') if t.strip()]
    c_tags = [t.strip() for t in str(cand_row.get('tags', '')).split(';') if t.strip()]
    common_tags = [t for t in c_tags if t in q_tags]

    all_contributions = []
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

    top_features = {"genres": [], "tags": []}
    for item in top_items:
        top_features[item['category']].append({
            "name": item['name'],
            "score": round(float(item['score']), 4),
            "pct": item['pct']
        })

    other_features = {"genres": [], "tags": []}
    for item in other_items:
        other_features[item['category']].append({
            "name": item['name'],
            "score": round(float(item['score']), 4),
            "pct": item['pct']
        })

    cand_name = str(cand_row.get('name', ''))
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


def get_corpus_word_count(row):
    """
    Menghitung jumlah kata pada corpus gabungan (deskripsi + genre + tag).
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


def check_is_sparse_corpus(row):
    """
    Mengecek apakah game termasuk kategori sparse corpus (deskripsi kosong / kata < 15).
    """
    empty_s = str(row.get('short_description', '')).strip() in ['', 'nan']
    empty_d = str(row.get('detailed_description', '')).strip() in ['', 'nan']
    both_empty = empty_s and empty_d
    word_cnt = get_corpus_word_count(row)
    return bool(both_empty or word_cnt < 15)


from functools import lru_cache

# ==============================================================================
# ALGORITMA REKOMENDASI UTAMA DENGAN IN-MEMORY LRU CACHE
# ==============================================================================
import time

@lru_cache(maxsize=256)
def _cached_get_recommendations(query_clean, top_n=12):
    t0 = time.time()
    
    matches = df[df['name'].str.lower() == query_clean]
    if matches.empty:
        matches = df[df['name'].str.lower().str.contains(query_clean, regex=False, na=False)]
        
    if matches.empty:
        return None, f"Game '{query_clean}' tidak ditemukan dalam sistem kami."

    target_idx = matches.index[0]
    target_row = df.iloc[target_idx]
    game_target_name = target_row['name']
    t_match = time.time()

    # 1. REAL-TIME COSINE SIMILARITY
    query_vec = tfidf_matrix[target_idx]
    sim_scores = linear_kernel(query_vec, tfidf_matrix).flatten()
    t_cosine = time.time()

    # 2. TIE-BREAKER & SOFT PENALTY (PRECOMPUTED NUMPY ARRAYS FOR HIGH-SPEED VECTORIZED ACCESS)
    is_sparse_arr = df['is_sparse_corpus'].values if 'is_sparse_corpus' in df.columns else np.zeros(len(df), dtype=bool)
    pos_rev_arr = df['positive_reviews'].values if 'positive_reviews' in df.columns else np.zeros(len(df), dtype=float)
    
    t_sparse_start = time.time()
    penalty_factors = np.where(is_sparse_arr, 0.90, 1.00)
    final_ranking_scores = sim_scores * penalty_factors
    t_sparse_sum = time.time() - t_sparse_start

    # Exclude target_idx by setting score to -1
    final_ranking_scores[target_idx] = -1.0

    # Fast multi-key sorting using numpy lexsort: secondary (pos_rev_arr), primary (final_ranking_scores)
    t_sort_start = time.time()
    sorted_indices = np.lexsort((pos_rev_arr, final_ranking_scores))[::-1]
    t_sort = time.time()

    candidates = []
    # Take top 100 candidate indices for diversification loop to avoid iterating 24k items
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
    accepted_recommendations = []
    accepted_titles = [game_target_name]

    t_xai_sum = 0
    t_lev_sum = 0

    for cand in candidates:
        cand_row = df.iloc[cand['index']]
        cand_name = cand_row['name']
        
        t_lev_start = time.time()
        is_duplicate_sequel = False
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
                target_idx, cand['index'], target_row, cand_row
            )
            t_xai_sum += (time.time() - t_xai_start)

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

    t_loop = time.time()

    if accepted_recommendations:
        max_sim_pct = max(r['similarity_percentage'] for r in accepted_recommendations)
        tight_count = sum(1 for r in accepted_recommendations if (max_sim_pct - r['similarity_percentage']) <= 1.5)
        is_tight_group = tight_count >= 3

        for r in accepted_recommendations:
            r['is_tight_cluster'] = bool(is_tight_group and (max_sim_pct - r['similarity_percentage']) <= 1.5)

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
        'is_sparse_corpus': check_is_sparse_corpus(target_row),
        'explanation': f"Game target pencarian utama."
    }
    t_end = time.time()

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

    popular_suggestions = ["Elden Ring", "Cyberpunk 2077", "The Witcher 3", "Palworld", "Baldur's Gate 3", "Grand Theft Auto V"]
    suggestions = None

    if search_query:
        result, error_msg = get_recommendations_data(search_query, top_n=top_n)
        if error_msg:
            error = error_msg
            suggestions = popular_suggestions
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
        target_game=target_game,
        recommendations=recommendations if target_game else None,
        suggestions=suggestions,
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


