from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import linear_kernel
import pickle
import pandas as pd
import os
import Levenshtein  # BATASAN PROPOSAL: Pustaka Levenshtein untuk Diversification Filtering

app = Flask(__name__)

# ==============================================================================
# BATASAN PROPOSAL: LAZY LOADING
# Memuat file .pkl ke dalam RAM peladen HANYA saat aplikasi Flask menyala
# (global scope, di luar fungsi route)
# ==============================================================================
print("[INFO] [LAZY LOADING] Memuat file model .pkl ke dalam RAM peladen...")
df = None
tfidf_matrix = None
tfidf_vectorizer = None
indices = None

try:
    df = pickle.load(open('models/game_data.pkl', 'rb'))
    tfidf_matrix = pickle.load(open('models/tfidf_matrix.pkl', 'rb'))
    tfidf_vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
    indices = pickle.load(open('models/indices.pkl', 'rb'))
    print("[SUCCESS] Model dan data sparse TF-IDF berhasil dimuat ke RAM peladen!")
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
    Rasio < 0.3 berarti kemiripan string > 70% (kemungkinan besar sekuel/versi ulang).
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


# ==============================================================================
# ALGORITMA REKOMENDASI UTAMA
# ==============================================================================
def get_recommendations_data(title, top_n=10):
    """
    Fungsi utama pengolah rekomendasi yang menerapkan:
    1. Real-Time Cosine Similarity
    2. Tie-Breaker (Positive Reviews)
    3. Diversification Filtering (Levenshtein Distance)
    4. Explainable AI (XAI) Template-Based Text Generation
    """
    if df is None or tfidf_matrix is None or indices is None:
        return None, "Model belum dimuat ke RAM."

    # Search Case-Insensitive
    matches = df[df['name'].str.lower() == title.lower().strip()]
    if matches.empty:
        return None, f"Game '{title}' tidak ditemukan dalam database."

    target_idx = matches.index[0]
    target_row = df.iloc[target_idx]
    game_target_name = target_row['name']

    # 1. REAL-TIME COSINE SIMILARITY
    # Menghitung skor kemiripan vektor query terhadap seluruh vektor matriks TF-IDF sparse
    query_vec = tfidf_matrix[target_idx]
    sim_scores = linear_kernel(query_vec, tfidf_matrix).flatten()

    # 2. TIE-BREAKER MECHANISM
    # Gabungkan (index, cosine_score, positive_reviews)
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
    accepted_titles = [game_target_name]  # Sertakan target game agar sekuel target disaring

    for cand in candidates:
        cand_row = df.iloc[cand['index']]
        cand_name = cand_row['name']
        
        # Cek jarak edit dengan game yang sudah diterima di daftar Top-N
        is_duplicate_sequel = False
        for acc_title in accepted_titles:
            edit_ratio = calc_edit_distance_ratio(cand_name, acc_title)
            # Jika edit distance ratio < 0.3 (kemiripan string > 70%), abaikan sekuel berulang ini
            if edit_ratio < 0.3:
                is_duplicate_sequel = True
                break

        if not is_duplicate_sequel:
            accepted_titles.append(cand_name)
            
            sim_percentage = round(cand['sim_score'] * 100, 1)
            dominant_genre = extract_dominant_genre(target_row['genres'], cand_row['genres'])
            
            # 4. EXPLAINABLE AI (XAI) TANPA LLM: TEMPLATE-BASED TEXT GENERATION
            # Template: "Game [nama_game] direkomendasikan karena memiliki kesamaan [genre_dominan] dengan skor [skor]% terhadap [game_target]."
            xai_explanation = (
                f"Game {cand_name} direkomendasikan karena memiliki kesamaan {dominant_genre} "
                f"dengan skor {sim_percentage}% terhadap {game_target_name}."
            )

            rec_item = {
                'steam_appid': int(cand_row['steam_appid']),
                'name': str(cand_name),
                'price': float(cand_row['price']),
                'genres': str(cand_row['genres']),
                'tags': str(cand_row['tags']),
                'header_image': str(cand_row['header_image']),
                'short_description': str(cand_row['short_description']),
                'detailed_description': str(cand_row['detailed_description']),
                'rating_score': float(cand_row['rating_score']),
                'rating': str(cand_row['rating']),
                'positive_reviews': float(cand_row['positive_reviews']),
                'total_reviews': float(cand_row['total_reviews']),
                'similarity_score': sim_percentage,
                'explanation': xai_explanation  # Teks XAI Statis berbasis Template
            }
            accepted_recommendations.append(rec_item)

        if len(accepted_recommendations) >= top_n:
            break

    # Format data game target
    target_data = {
        'steam_appid': int(target_row['steam_appid']),
        'name': str(target_row['name']),
        'price': float(target_row['price']),
        'genres': str(target_row['genres']),
        'tags': str(target_row['tags']),
        'header_image': str(target_row['header_image']),
        'short_description': str(target_row['short_description']),
        'detailed_description': str(target_row['detailed_description']),
        'rating_score': float(target_row['rating_score']),
        'rating': str(target_row['rating']),
        'positive_reviews': float(target_row['positive_reviews']),
        'total_reviews': float(target_row['total_reviews']),
        'similarity_score': 100.0,
        'explanation': f"Game target pencarian utama."
    }

    return {
        'target_game': target_data,
        'recommendations': accepted_recommendations
    }, None


# ==============================================================================
# ROUTING API & WEB
# ==============================================================================

# BATASAN PROPOSAL: Endpoint API JSON untuk Frontend HTML/JS
@app.route('/api/recommend', methods=['GET', 'POST'])
def api_recommend():
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
        'actual_title': result['target_game']['name'],
        'target_game': result['target_game'],
        'recommendations': result['recommendations']
    })


@app.route('/', methods=['GET', 'POST'])
def home():
    query_param = request.args.get('q', '')
    wants_json = (
        request.args.get('format') == 'json' or 
        request.is_json or 
        'application/json' in request.headers.get('Accept', '')
    )
    
    if request.method == 'POST':
        game_title = request.form.get('game_title', '').strip()
    else:
        game_title = query_param.strip()

    if game_title:
        result, error_msg = get_recommendations_data(game_title)
        
        # Jika client meminta format JSON
        if wants_json:
            if error_msg:
                return jsonify({'status': 'error', 'message': error_msg}), 404
            return jsonify({
                'status': 'success',
                'actual_title': result['target_game']['name'],
                'target_game': result['target_game'],
                'recommendations': result['recommendations']
            })

        if result is not None:
            # Konversi data ke DataFrame agar kompatibel dengan template index.html Jinja2 existing
            target_df = pd.DataFrame([result['target_game']])
            rec_df = pd.DataFrame(result['recommendations'])
            final_df = pd.concat([target_df, rec_df], ignore_index=True)
            
            return render_template(
                'index.html', 
                recommendations=final_df, 
                actual_title=result['target_game']['name'], 
                search_query=result['target_game']['name']
            )

        # Jika game tidak ditemukan
        try:
            suggestions = df.sort_values(by='rating_score', ascending=False)['name'].head(50).sample(5).tolist()
        except Exception:
            suggestions = ["Elden Ring", "Cyberpunk 2077", "Stardew Valley", "Terraria", "Portal 2"]
            
        return render_template('index.html', error=error_msg, search_query=game_title, suggestions=suggestions)

    return render_template('index.html')


@app.route('/api/search_autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('term', '').lower().strip()
    if not query or df is None:
        return jsonify([])
    matches = df[df['name'].str.lower().str.contains(query, na=False)]['name'].head(10).tolist()
    return jsonify(matches)


if __name__ == '__main__':
    # Memilih port 5000 untuk pengujian lokal
    app.run(debug=True, port=5000)