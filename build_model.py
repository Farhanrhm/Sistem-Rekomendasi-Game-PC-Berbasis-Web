import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

os.makedirs('models', exist_ok=True)

print("Membaca dataset...")
data_path = 'dataset/processed/steam_new_and_fav_final_1.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File dataset {data_path} tidak ditemukan. Jalankan smart_scraper.py terlebih dahulu.")

df = pd.read_csv(data_path)

# Pastikan kolom opsional/wajib ada
for col in ['clean_desc', 'detailed_description', 'genres', 'tags', 'short_description']:
    if col not in df.columns:
        df[col] = ''
    df[col] = df[col].fillna('')

for col in ['positive_reviews', 'total_reviews', 'rating_score', 'price']:
    if col not in df.columns:
        df[col] = 0
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# 1. GABUNGKAN FITUR TEKS
# Menggabungkan clean_desc / detailed_description dengan genres dan tags untuk fitur komprehensif
def combine_features(row):
    desc = str(row['clean_desc']) if str(row['clean_desc']).strip() else str(row['detailed_description'])
    genres = str(row['genres']).replace(';', ' ')
    tags = str(row['tags']).replace(';', ' ')
    return f"{desc} {genres} {tags}"

df['combined_features'] = df.apply(combine_features, axis=1)

print("Melakukan ekstraksi fitur teks gabungan dengan TfidfVectorizer (Scikit-Learn)...")
# TfidfVectorizer dengan stop_words english
tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
# Hasilkan matriks sparse (scipy.sparse.csr_matrix)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

print(f"Bentuk Matriks TF-IDF Sparse: {tfidf_matrix.shape}")
print(f"Tipe Matriks: {type(tfidf_matrix)}")

print("Menyimpan file model .pkl ke folder 'models'...")
# Simpan matriks sparse TF-IDF
with open('models/tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)

# Simpan model vectorizer
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Simpan Dataframe yang sudah dibersihkan
cols_to_keep = [
    'steam_appid', 'name', 'price', 'genres', 'header_image', 
    'short_description', 'detailed_description', 'rating', 
    'positive_reviews', 'total_reviews', 'rating_score', 'tags'
]

# Pastikan semua kolom yang diperlukan UI & algoritma ada di DataFrame
for col in cols_to_keep:
    if col not in df.columns:
        if col in ['positive_reviews', 'total_reviews', 'rating_score', 'price']:
            df[col] = 0
        else:
            df[col] = ''

df_clean = df[cols_to_keep].copy()

with open('models/game_data.pkl', 'wb') as f:
    pickle.dump(df_clean, f)

# Simpan mapping indeks judul game
indices = pd.Series(df_clean.index, index=df_clean['name'].str.lower()).drop_duplicates()
with open('models/indices.pkl', 'wb') as f:
    pickle.dump(indices, f)

print("[OK] --- PROSES PEMODELAN IN-MEMORY (.PKL) SELESAI ---")