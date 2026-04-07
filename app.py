from flask import Flask, render_template, request, jsonify
from functools import lru_cache
import pickle
import pandas as pd
import os

app = Flask(__name__)

# --- LOAD MODEL & DATA ---
print("Memuat model AI ke dalam memori...")
try:
    df = pickle.load(open('models/game_data.pkl', 'rb'))
    cosine_sim = pickle.load(open('models/cosine_sim.pkl', 'rb'))
    indices = pickle.load(open('models/indices.pkl', 'rb'))
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error memuat model: {e}")
    print("Pastikan Anda sudah menjalankan 3_build_model.py")

# --- LOGIKA REKOMENDASI (DENGAN CACHING) ---
@lru_cache(maxsize=100)
def get_recommendations(title):
    try:
        idx = indices[title]
        
        # Jika ada judul duplikat, ambil yang pertama
        if type(idx) == pd.Series:
            idx = idx.iloc[0]

        # Hitung skor kemiripan
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Ambil 12 game teratas (index 1-12 karena index 0 adalah game itu sendiri)
        sim_scores = sim_scores[1:13]

        game_indices = [i[0] for i in sim_scores]
        similarity_percentages = [i[1] * 100 for i in sim_scores]

        # Buat dataframe untuk rekomendasi
        rec_df = df.iloc[game_indices].copy()
        rec_df['similarity_score'] = similarity_percentages

        # Masukkan game utama di paling atas (untuk ditampilkan di Spotlight)
        main_game = df.iloc[[idx]].copy()
        main_game['similarity_score'] = 100.0

        # Gabungkan
        final_df = pd.concat([main_game, rec_df])
        return final_df
    except KeyError:
        return None

# --- ROUTING WEB ---
@app.route('/', methods=['GET', 'POST'])
def home():
    # Menangani parameter (Fitur Share Link)
    query_param = request.args.get('q', '')
    
    if request.method == 'POST':
        game_title = request.form.get('game_title', '').strip()
    else:
        game_title = query_param.strip()

    if game_title:
        # Pencarian Case-Insensitive
        actual_title_array = df[df['name'].str.lower() == game_title.lower()]['name'].values
        if len(actual_title_array) > 0:
            actual_title = actual_title_array[0]
            # Panggil fungsi yang sudah di-cache
            recommendations = get_recommendations(actual_title)
            
            if recommendations is not None:
                return render_template('index.html', recommendations=recommendations, actual_title=actual_title, search_query=actual_title)
        
        # Fitur Fallback: Game tidak ditemukan
        error_msg = f"Game '{game_title}' tidak ditemukan di database kami."
        # Ambil 5 game populer (berdasarkan rating) acak sebagai saran
        try:
            popular_games = df.sort_values(by='rating_score', ascending=False)['name'].head(50).sample(5).tolist()
        except:
            popular_games = []
        return render_template('index.html', error=error_msg, search_query=game_title, suggestions=popular_games)

    return render_template('index.html')

@app.route('/api/search_autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('term', '').lower()
    if not query:
        return jsonify([])
    # Cari 10 game yang mengandung kata kunci
    matches = df[df['name'].str.lower().str.contains(query, na=False)]['name'].head(10).tolist()
    return jsonify(matches)

if __name__ == '__main__':
    # Hapus debug=True saat skripsi sudah disidangkan/di-deploy
    app.run(debug=True, port=5000)