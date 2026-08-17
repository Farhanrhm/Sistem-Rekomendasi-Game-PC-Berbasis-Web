import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import re

# Memastikan direktori penyimpan model siap
os.makedirs('models', exist_ok=True)

print("[INFO] Membaca dataset game...")
dataset_path = 'dataset/processed/steam_new_and_fav.csv'

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"File dataset {dataset_path} tidak ditemukan. Harap jalankan scraper/preprocessing terlebih dahulu.")

df = pd.read_csv(dataset_path)

# ==============================================================================
# PREPROCESSING TEKS & PEMBENTUKAN FITUR GABUNGAN
# ==============================================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]*>', ' ', text)  # Hapus tag HTML
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Hapus karakter spesial
    return text.lower().strip()

# Memastikan kolom teks memiliki nilai string default
if 'clean_desc' not in df.columns or df['clean_desc'].isnull().all():
    detailed = df['detailed_description'].fillna('') if 'detailed_description' in df.columns else ''
    short = df['short_description'].fillna('') if 'short_description' in df.columns else ''
    df['clean_desc'] = (detailed + " " + short).apply(clean_text)
else:
    df['clean_desc'] = df['clean_desc'].fillna('')

df['genres'] = df['genres'].fillna('').astype(str)
df['tags'] = df['tags'].fillna('').astype(str)

# BATASAN PROPOSAL: "Ekstraksi fitur teks gabungan"
# Menggabungkan atribut teks (deskripsi bersih, genres, dan tags) menjadi satu dokumen fitur teks
df['combined_features'] = (
    df['clean_desc'] + " " +
    df['genres'].str.replace(';', ' ') + " " +
    df['tags'].str.replace(';', ' ')
)

print("[INFO] Ekstraksi fitur menggunakan TfidfVectorizer (Scikit-Learn)...")
# BATASAN PROPOSAL: Ekstraksi fitur teks gabungan menggunakan TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

print(f"[INFO] Ukuran Matriks Sparse TF-IDF: {tfidf_matrix.shape} (Baris: Game, Kolom: Fitur Kata)")

# ==============================================================================
# PENYIAPAN COLUMNS DATAFRAME & METRIK
# ==============================================================================
cols_to_keep = [
    'steam_appid', 'name', 'price', 'genres', 'header_image', 
    'short_description', 'detailed_description', 'rating_score', 
    'rating', 'positive_reviews', 'total_reviews', 'clean_desc', 'tags'
]

for col in cols_to_keep:
    if col not in df.columns:
        if col in ['rating_score', 'price', 'positive_reviews', 'total_reviews']:
            df[col] = 0.0
        else:
            df[col] = ''
    
    if col in ['rating_score', 'price', 'positive_reviews', 'total_reviews']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

# Menghitung estimasi rating_score jika belum terisi
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

df['rating_score'] = df.apply(estimate_score, axis=1)

# ==============================================================================
# ARSITEKTUR IN-MEMORY STORAGE (MEMORI SANGAT RINGAN 50-80 MB)
# BATASAN PROPOSAL:
# Menyimpan hasil TF-IDF berupa matriks sparse, vectorizer, dan dataframe bersih
# ke dalam format Pickle (.pkl) agar estimasi penggunaan RAM server sangat ringan.
# ==============================================================================
print("[INFO] Menyimpan model dan data ke format Pickle (.pkl)...")

# 1. Simpan DataFrame Bersih
df_clean = df[cols_to_keep].copy()
pickle.dump(df_clean, open('models/game_data.pkl', 'wb'))

# 2. Simpan Matriks Sparse TF-IDF (Bukan matriks kemiripan dense 15k x 15k yang berat)
pickle.dump(tfidf_matrix, open('models/tfidf_matrix.pkl', 'wb'))

# 3. Simpan Model TfidfVectorizer
pickle.dump(tfidf_vectorizer, open('models/tfidf_vectorizer.pkl', 'wb'))

# 4. Simpan Indeks Pemetaan Judul -> Indeks
indices = pd.Series(df_clean.index, index=df_clean['name']).drop_duplicates()
pickle.dump(indices, open('models/indices.pkl', 'wb'))

print("[SUCCESS] --- MODEL DAN METADATA TF-IDF SPARSE BERHASIL DIBUAT ---")