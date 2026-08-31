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

# 1. NOISE REDUCTION: FILTER PLATFORM & ACCESSIBILITY NON-CONTENT TAGS
PLATFORM_FEATURE_TAGS = {
    # Platform & Licensing Tags
    "steam cloud", "steam achievements", "steam trading cards", 
    "surround sound", "stereo sound", "cross-platform multiplayer", 
    "steam workshop", "full controller support", "partial controller support", 
    "remote play on phone", "remote play on tablet", "remote play on tv", 
    "remote play together", "steam turn-notifications", "steam leaderboards", 
    "captions available", "commentary available", "includes level editor", 
    "hdr available", "tracked controller support", "family sharing",
    # Technical & Accessibility Non-Content Tags
    "custom volume controls", "playable without timed input", "save anytime", 
    "mouse only option", "keyboard only option", "touch only option", 
    "camera comfort", "adjustable difficulty", "adjustable text size", 
    "subtitle options", "dualshock controller support", "dualsense controller support", 
    "color alternatives", "stats", "shared/split screen", "vr only", 
    "gamepad recommended", "in-app purchases", "shared/split screen pvp", 
    "shared/split screen co-op", "#category_playable_at_your_own_pace"
}

def clean_platform_tags(tags_str):
    if not tags_str or not isinstance(tags_str, str):
        return ""
    tags = [t.strip() for t in tags_str.split(';') if t.strip()]
    pure_tags = [t for t in tags if t.lower() not in PLATFORM_FEATURE_TAGS]
    return "; ".join(pure_tags)

df['tags'] = df['tags'].apply(clean_platform_tags)

import re

def clean_html_and_urls(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.lower().strip()

# GABUNGKAN FITUR TEKS
def combine_features(row):
    raw_desc = str(row['short_description']) + " " + str(row['detailed_description'])
    desc = clean_html_and_urls(raw_desc)
    genres = str(row['genres']).replace(';', ' ') if str(row['genres']) != 'nan' else ""
    tags = str(row['tags']).replace(';', ' ') if str(row['tags']) != 'nan' else ""
    return f"{desc} {genres} {tags}"

df['combined_features'] = df.apply(combine_features, axis=1)

print("Melakukan ekstraksi fitur teks gabungan dengan TfidfVectorizer (Scikit-Learn)...")
# TfidfVectorizer dengan stop_words english & max_df=0.50 untuk mengeliminasi term universal super-high DF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.50)
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

def check_is_sparse(row):
    short_d = str(row.get('short_description', '')).strip()
    detail_d = str(row.get('detailed_description', '')).strip()
    both_empty = (short_d in ['', 'nan']) and (detail_d in ['', 'nan'])
    
    desc_text = short_d if (short_d and short_d != 'nan') else detail_d
    if desc_text == 'nan': desc_text = ""
    genres = str(row.get('genres', '')).replace(';', ' ') if str(row.get('genres', '')).strip() != 'nan' else ""
    tags = str(row.get('tags', '')).replace(';', ' ') if str(row.get('tags', '')).strip() != 'nan' else ""
    full_text = f"{desc_text} {genres} {tags}".strip()
    word_cnt = len([w for w in full_text.split() if w])
    return bool(both_empty or word_cnt < 15)

df['is_sparse_corpus'] = df.apply(check_is_sparse, axis=1)

# Simpan Dataframe yang sudah dibersihkan
cols_to_keep = [
    'steam_appid', 'name', 'price', 'genres', 'header_image', 
    'short_description', 'detailed_description', 'rating', 
    'positive_reviews', 'total_reviews', 'rating_score', 'tags', 'is_sparse_corpus'
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