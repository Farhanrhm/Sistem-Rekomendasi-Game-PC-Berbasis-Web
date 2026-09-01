import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import app

print("Menyiapkan dataset dan environment...")
df = app.df
print(f"Total game dalam dataset: {len(df)}")

# Pastikan seed reproducible
np.random.seed(42)

# BAGIAN 1: Genre-Overlap Consistency Score (Sample 400 game)
sample_size = 400
sample_indices = np.random.choice(df.index, size=sample_size, replace=False)
sample_games = df.loc[sample_indices]

def calc_jaccard_overlap(str1, str2):
    set1 = set([g.strip().lower() for g in str(str1).split(';') if g.strip()])
    set2 = set([g.strip().lower() for g in str(str2).split(';') if g.strip()])
    if not set1 or not set2:
        return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

results_list = []

for idx, q_row in sample_games.iterrows():
    q_name = q_row['name']
    rec_res, err = app.get_recommendations_data(q_name, top_n=5)
    if err or not rec_res:
        continue
    
    q_genres = q_row['genres']
    q_is_sparse = bool(q_row.get('is_sparse_corpus', False))
    
    for rank, rec in enumerate(rec_res['recommendations'][:5], 1):
        rec_genres = rec['genres']
        rec_sim_pct = rec['similarity_percentage']
        rec_is_sparse = rec['is_sparse_corpus']
        
        jaccard_sim = calc_jaccard_overlap(q_genres, rec_genres)
        
        results_list.append({
            'query_name': q_name,
            'query_is_sparse': q_is_sparse,
            'rec_rank': rank,
            'rec_name': rec['name'],
            'rec_sim_pct': rec_sim_pct,
            'rec_is_sparse': rec_is_sparse,
            'genre_overlap_ratio': jaccard_sim
        })

df_res = pd.DataFrame(results_list)

print(f"\n=======================================================")
print(f"HASIL EVALUASI BAGIAN 1: GENRE-OVERLAP CONSISTENCY SCORE")
print(f"=======================================================")
print(f"Total pasangan (Query -> Rec Top-5) dievaluasi: {len(df_res)}")

overall_mean_overlap = df_res['genre_overlap_ratio'].mean()
print(f"1. Rata-rata Genre Overlap Ratio Keseluruhan: {overall_mean_overlap:.4f} ({overall_mean_overlap*100:.2f}%)")

# Segmentasi Non-Sparse vs Sparse
non_sparse_df = df_res[~df_res['query_is_sparse']]
sparse_df = df_res[df_res['query_is_sparse']]

print(f"\n2. Breakdown Kategori Deskripsi Game:")
print(f"   - Non-Sparse Query ({len(non_sparse_df)} pasang): Mean Overlap = {non_sparse_df['genre_overlap_ratio'].mean():.4f} ({non_sparse_df['genre_overlap_ratio'].mean()*100:.2f}%)")
print(f"   - Sparse Query ({len(sparse_df)} pasang)    : Mean Overlap = {sparse_df['genre_overlap_ratio'].mean():.4f} ({sparse_df['genre_overlap_ratio'].mean()*100:.2f}%)")

# Segmentasi Berdasarkan Rentang Similarity Score
high_sim = df_res[df_res['rec_sim_pct'] >= 80.0]
mid_sim = df_res[(df_res['rec_sim_pct'] >= 50.0) & (df_res['rec_sim_pct'] < 80.0)]
low_sim = df_res[df_res['rec_sim_pct'] < 50.0]

print(f"\n3. Breakdown Rentang Skor Kemiripan (Similarity Percentage):")
print(f"   - High Similarity (80% - 100%, n={len(high_sim)}): Mean Overlap = {high_sim['genre_overlap_ratio'].mean():.4f} ({high_sim['genre_overlap_ratio'].mean()*100:.2f}%)")
print(f"   - Mid Similarity  (50% - 79.9%, n={len(mid_sim)}): Mean Overlap = {mid_sim['genre_overlap_ratio'].mean():.4f} ({mid_sim['genre_overlap_ratio'].mean()*100:.2f}%)")
print(f"   - Low Similarity  (< 50%, n={len(low_sim)})     : Mean Overlap = {low_sim['genre_overlap_ratio'].mean():.4f} ({low_sim['genre_overlap_ratio'].mean()*100:.2f}%)")

# Pearson Correlation Coefficient
corr, p_val = pearsonr(df_res['rec_sim_pct'], df_res['genre_overlap_ratio'])
print(f"\n4. Korelasi Pearson (Similarity % vs Genre Overlap Ratio):")
print(f"   - Pearson Correlation Coefficient (r) = {corr:.4f}")
print(f"   - p-value = {p_val:.4e}")

# ==============================================================================
# BAGIAN 2: Persiapan Sample Evaluasi Manual (18 Game Query)
# ==============================================================================
print(f"\n=======================================================")
print(f"HASIL EVALUASI BAGIAN 2: PERSIAPAN SAMPLE EVALUASI MANUAL")
print(f"=======================================================")

top_popular = df[~df['is_sparse_corpus']].sort_values(by='positive_reviews', ascending=False)
popular_sample = top_popular.iloc[[0, 10, 25, 40, 75, 120]]

mid_tier = df[(~df['is_sparse_corpus']) & (df['positive_reviews'] > 100) & (df['positive_reviews'] < 2000)]
mid_sample = mid_tier.sample(n=6, random_state=42)

sparse_tier = df[df['is_sparse_corpus']]
sparse_sample = sparse_tier.sample(n=6, random_state=42)

eval_queries = pd.concat([popular_sample, mid_sample, sparse_sample])

manual_eval_rows = []
counter = 1

for idx, q_row in eval_queries.iterrows():
    q_name = q_row['name']
    q_genres = q_row['genres']
    q_sparse_str = "Sparse" if q_row['is_sparse_corpus'] else "Lengkap"
    
    rec_res, err = app.get_recommendations_data(q_name, top_n=5)
    if err or not rec_res:
        continue
        
    for rec in rec_res['recommendations'][:5]:
        manual_eval_rows.append({
            'No': counter,
            'Game Query': q_name,
            'Status Query': q_sparse_str,
            'Genre Query': q_genres,
            'Game Rekomendasi': rec['name'],
            'Similarity %': rec['similarity_percentage'],
            'Genre Rekomendasi': rec['genres'],
            'Tags Rekomendasi': rec['tags'],
            'Status Rec': "Sparse" if rec['is_sparse_corpus'] else "Lengkap",
            'Skor Relevansi (1-5, diisi manual)': "",
            'Catatan (diisi manual)': ""
        })
        counter += 1

manual_df = pd.DataFrame(manual_eval_rows)
out_csv = os.path.join(os.getcwd(), "evaluasi_relevansi_manual_levelfind.csv")
manual_df.to_csv(out_csv, index=False, encoding='utf-8-sig')

print(f"File evaluasi manual berhasil dibuat: {out_csv}")
print(f"Total pasangan query-rekomendasi untuk dinilai manual: {len(manual_df)} baris (dari {len(eval_queries)} game query).\n")
