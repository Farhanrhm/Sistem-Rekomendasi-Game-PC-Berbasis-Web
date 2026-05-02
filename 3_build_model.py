import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle
import os

os.makedirs('models', exist_ok=True)

print("Membaca data...")
df = pd.read_csv('dataset/processed/dataset_final_skripsi.csv')

# Pastikan tidak ada data yang kosong (NaN) pada kolom fitur
df['clean_desc'] = df['clean_desc'].fillna('')
df['genres'] = df['genres'].fillna('')
df['tags'] = df['tags'].fillna('')

print("Menghitung matriks TF-IDF untuk masing-masing elemen...")

# --- 1. MODEL TF-IDF: DESKRIPSI (Bobot 50%) ---
tfidf_desc = TfidfVectorizer(stop_words='english')
tfidf_matrix_desc = tfidf_desc.fit_transform(df['clean_desc'])
sim_desc = linear_kernel(tfidf_matrix_desc, tfidf_matrix_desc)

# --- 2. MODEL TF-IDF: GENRE (Bobot 30%) ---
tfidf_genre = TfidfVectorizer(stop_words='english')
tfidf_matrix_genre = tfidf_genre.fit_transform(df['genres'])
sim_genre = linear_kernel(tfidf_matrix_genre, tfidf_matrix_genre)

# --- 3. MODEL TF-IDF: TAGS STEAM (Bobot 20%) ---
tfidf_tag = TfidfVectorizer(stop_words='english')
tfidf_matrix_tag = tfidf_tag.fit_transform(df['tags'])
sim_tag = linear_kernel(tfidf_matrix_tag, tfidf_matrix_tag)

print("Mengkalkulasi Pembobotan (Weighted Cosine Similarity)...")
# Rumus Matematika: Sim_Total = (0.5 * Sim_Desc) + (0.3 * Sim_Genre) + (0.2 * Sim_Tag)
cosine_sim = (0.50 * sim_desc) + (0.30 * sim_genre) + (0.20 * sim_tag)

print("Menyimpan model ke folder 'models'...")
# Daftar kolom yang dibutuhkan oleh Web HTML
cols_to_keep = ['steam_appid', 'name', 'price', 'genres', 'header_image', 'short_description', 'detailed_description', 'rating_score', 'rating', 'total_reviews', 'clean_desc', 'tags']

# Pastikan semua kolom tersedia dan bertipe data yang benar (Mencegah TypeError)
for col in cols_to_keep:
    if col not in df.columns:
        if col in ['rating_score', 'price']:
            df[col] = 0.0
        else:
            df[col] = ''
    
    if col in ['rating_score', 'price']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

# Mapping sederhana jika rating_score masih 0 tapi rating (string) sudah ada
# Ini membantu data lama agar tetap tampil bintangnya di UI
def estimate_score(row):
    if row['rating_score'] > 0:
        return row['rating_score']
    
    mapping = {
        'Overwhelmingly Positive': 95,
        'Very Positive': 85,
        'Mostly Positive': 75,
        'Positive': 80,
        'Mixed': 50,
        'Mostly Negative': 30,
        'Very Negative': 15,
        'Overwhelmingly Negative': 10
    }
    return mapping.get(row['rating'], 0)

if 'rating' in df.columns:
    df['rating_score'] = df.apply(estimate_score, axis=1)

# Simpan Data dan Model Matriks yang baru
pickle.dump(df[cols_to_keep], open('models/game_data.pkl', 'wb'))
pickle.dump(cosine_sim, open('models/cosine_sim.pkl', 'wb'))

indices = pd.Series(df.index, index=df['name']).drop_duplicates()
pickle.dump(indices, open('models/indices.pkl', 'wb'))

print("--- MODEL BERHASIL DIBUAT ---")