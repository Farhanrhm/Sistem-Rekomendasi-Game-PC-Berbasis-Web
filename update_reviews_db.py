import os
import sys
import time
import shutil
import requests
import numpy as np
import pandas as pd

# --- KONFIGURASI FILE & API ---
INPUT_FILE = 'dataset/processed/steam_new_and_fav.csv'
BACKUP_FILE = 'dataset/processed/steam_new_and_fav_backup.csv'

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

def make_backup():
    """Membuat salinan cadangan (backup) file CSV asli sebelum proses dimulai."""
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] File database {INPUT_FILE} tidak ditemukan. Tidak dapat membuat backup.")
        sys.exit(1)
    
    print(f"[BACKUP] Membuat salinan cadangan database ke {BACKUP_FILE}...")
    try:
        shutil.copyfile(INPUT_FILE, BACKUP_FILE)
        print("[OK] Backup berhasil dibuat.")
    except Exception as e:
        print(f"[ERROR] Gagal membuat backup: {e}")
        sys.exit(1)

def fetch_steam_reviews(app_id):
    """
    Mengambil data ulasan dari Steam Reviews API dengan penanganan Rate Limit (429)
    dan retry mechanism yang tangguh.
    """
    url = f"https://store.steampowered.com/appreviews/{app_id}"
    params = {
        'json': '1',
        'language': 'english',
        'purchase_type': 'all'
    }
    
    max_retries = 5
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            res = requests.get(url, params=params, headers=HEADERS, timeout=15)
            
            # Deteksi Rate Limit (429)
            if res.status_code == 429:
                print(f"\n[WARNING] [Rate Limit 429] Terdeteksi untuk AppID {app_id}. Menunggu 60 detik sebelum mencoba kembali...")
                time.sleep(60)
                continue
                
            if res.status_code != 200:
                print(f"\n[WARNING] HTTP {res.status_code} diterima untuk AppID {app_id}. Mencoba lagi dalam {retry_delay} detik...")
                time.sleep(retry_delay)
                continue
                
            data = res.json()
            if 'query_summary' in data:
                return data['query_summary']
            else:
                return None
                
        except requests.RequestException as e:
            print(f"\n[WARNING] Koneksi error untuk AppID {app_id}: {e}. Mencoba lagi dalam {retry_delay} detik...")
            time.sleep(retry_delay)
            
    return None

def save_dataframe(df, filepath):
    """Menyimpan DataFrame ke file CSV dengan tipe data yang rapi."""
    # Konversi kolom integer nullable agar tidak menghasilkan float .0 di CSV
    df_save = df.copy()
    try:
        df_save['total_reviews'] = df_save['total_reviews'].astype('Int64')
        df_save['positive_reviews'] = df_save['positive_reviews'].astype('Int64')
    except Exception:
        pass
    
    df_save.to_csv(filepath, index=False)

def main():
    # 1. Jalankan Auto-Backup
    make_backup()
    
    # 2. Muat File CSV menggunakan Pandas
    print(f"[READ] Membaca database CSV {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Bersihkan baris yang tidak memiliki steam_appid valid
    df = df.dropna(subset=['steam_appid'])
    df['steam_appid'] = df['steam_appid'].astype(int)
    
    # Inisialisasi kolom-kolom baru jika belum ada di CSV asli
    for col in ['total_reviews', 'positive_reviews', 'rating_score']:
        if col not in df.columns:
            df[col] = np.nan
            
    if 'rating' not in df.columns:
        df['rating'] = np.nan
        
    if 'review_status' not in df.columns:
        df['review_status'] = np.nan
        
    # Pastikan tipe kolom teks adalah object, bukan float64 (agar terhindar dari FutureWarning)
    df['review_status'] = df['review_status'].astype(object)
    df['rating'] = df['rating'].astype(object)
        
    # 3. Filter game yang ulasannya kosong (NaN), bernilai 0, atau tidak ada
    # Kriteria belum dicek: review_status tidak bernilai 'checked'
    mask_not_checked = df['review_status'].isna() | (df['review_status'] != 'checked')
    
    # Kriteria ulasan kosong/0/tidak ada
    mask_empty_reviews = (
        df['total_reviews'].isna() | 
        (df['total_reviews'] == 0) | 
        df['rating'].isna() | 
        (df['rating'] == '') | 
        (df['rating'] == 'Unknown') | 
        (df['rating'] == 'N/A')
    )
    
    indices_to_update = df[mask_not_checked & mask_empty_reviews].index.tolist()
    total_to_update = len(indices_to_update)
    
    if total_to_update == 0:
        print("[OK] Semua game di database sudah memiliki data ulasan lengkap atau sudah diperiksa!")
        print("Tidak ada yang perlu diperbarui.")
        sys.exit(0)
        
    print(f"[FIND] Ditemukan {total_to_update} game yang ulasannya kosong/perlu diperbarui.")
    print("[START] Memulai proses scraping. Tekan Ctrl + C kapan saja untuk menghentikan dengan aman.")
    
    updated_count = 0
    
    # 4. Looping & Update
    try:
        for idx in indices_to_update:
            row = df.loc[idx]
            app_id = int(row['steam_appid'])
            game_name = row['name']
            
            # Tarik data dari Steam API
            summary = fetch_steam_reviews(app_id)
            
            if summary is not None:
                tot = summary.get('total_reviews', 0)
                pos = summary.get('total_positive', 0)
                rating_desc = summary.get('review_score_desc', 'No Rating')
                
                # Tangani game yang memang belum memiliki ulasan di Steam
                if tot == 0:
                    rating_desc = "No user reviews"
                    pos = 0
                    rating_score = 0.0
                else:
                    rating_score = round((pos / tot) * 100, 1)
                    
                # Update baris data
                df.at[idx, 'total_reviews'] = tot
                df.at[idx, 'positive_reviews'] = pos
                df.at[idx, 'rating'] = rating_desc
                df.at[idx, 'rating_score'] = rating_score
                df.at[idx, 'review_status'] = 'checked'
                
                updated_count += 1
                print(f"Updated [{updated_count}/{total_to_update}]: '{game_name}' -> rating: {rating_desc}, positive: {pos}, total: {tot}, score: {rating_score}%")
            else:
                # Gagal menarik data, lewati (biarkan review_status kosong agar bisa dicoba lagi nanti)
                print(f"[FAIL] Gagal mendapatkan data untuk '{game_name}' (AppID: {app_id})")
                
            # Auto-save berkala per 20 game
            if updated_count > 0 and updated_count % 20 == 0:
                save_dataframe(df, INPUT_FILE)
                print(f"[SAVE] [Auto-Save] Kemajuan berhasil disimpan ke {INPUT_FILE}")
                
            # Jeda 1.5 detik agar aman dari rate limit
            time.sleep(1.5)
            
    except KeyboardInterrupt:
        print("\n[STOP] Proses dihentikan oleh pengguna (KeyboardInterrupt).")
        
    finally:
        # 5. Simpan Hasil Akhir
        if updated_count > 0:
            print(f"[SAVE] Menyimpan hasil akhir ke {INPUT_FILE}...")
            save_dataframe(df, INPUT_FILE)
            print(f"[OK] Berhasil memperbarui {updated_count} game.")
        else:
            print("INFO: Tidak ada data baru yang diperbarui untuk disimpan.")
            
        print("\n========================================================")
        print("PENGINGAT PENTING:")
        print("Silakan jalankan kembali perintah berikut agar server membaca ulasan terbaru:")
        print("  python 3_build_model.py")
        print("========================================================")

if __name__ == '__main__':
    main()
