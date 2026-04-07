import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle
import os

os.makedirs('models', exist_ok=True)

print("Membaca data untuk model AI...")
df = pd.read_csv('dataset/processed/steam_ready_for_model.csv')

print("Menghitung matriks TF-IDF...")
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['soup'].fillna(''))

print("Menghitung skor Cosine Similarity...")
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

print("Menyimpan model ke folder 'models'...")
# Kolom wajib yang akan dikirim ke tampilan Web (HTML)
cols_to_keep = ['steam_appid', 'name', 'price', 'genres', 'header_image', 'short_description', 'detailed_description', 'rating_score']

for col in cols_to_keep:
    if col not in df.columns:
        df[col] = ''

# Ekspor data dan model
pickle.dump(df[cols_to_keep], open('models/game_data.pkl', 'wb'))
pickle.dump(cosine_sim, open('models/cosine_sim.pkl', 'wb'))

indices = pd.Series(df.index, index=df['name']).drop_duplicates()
pickle.dump(indices, open('models/indices.pkl', 'wb'))

print("✅ --- MODEL BERHASIL DIBUAT ---")