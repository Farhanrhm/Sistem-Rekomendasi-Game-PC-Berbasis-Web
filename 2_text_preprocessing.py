import pandas as pd
import re
import os

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.lower().strip()

print("Mulai Preprocessing Dataset game...")
input_file = 'dataset/processed/steam_new_and_fav.csv'
output_file = 'dataset/processed/steam_new_and_fav.csv'

if os.path.exists(input_file):
    df = pd.read_csv(input_file)
    print(f"Memproses {len(df)} game...")

    # Gabungkan deskripsi
    df['combined_desc'] = df['detailed_description'].fillna('') + " " + df['short_description'].fillna('')
    df['clean_desc'] = df['combined_desc'].apply(clean_text)
    df['soup'] = df['name'] + " " + df['tags'].fillna('') + " " + df['genres'].fillna('') + " " + df['clean_desc']
    
    # Buat SOUP: gabungan tags, genres, dan deskripsi bersih untuk diproses sistem
    df['soup'] = df['tags'].fillna('') + " " + df['genres'].fillna('') + " " + df['clean_desc']
    
    # Amankan kolom rating_score
    if 'rating_score' in df.columns:
        df['rating_score'] = df['rating_score'].fillna(0)
        
    df.to_csv(output_file, index=False)
    print(f"✅ Selesai! File siap di: {output_file}")
else:
    print("❌ Error: File CSV sumber tidak ditemukan!")