import requests
import pandas as pd
import time
import os
import re
import sys

# ==============================================================================
# KONFIGURASI UTAMA & BATASAN PROPOSAL
# BATASAN PROPOSAL: 'Purposive Sampling' membatasi maksimal 15.000 judul game.
# ==============================================================================
MAX_GAMES_LIMIT = 15000 
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}
OUTPUT_DIR = 'dataset/processed'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'steam_new_and_fav.csv')

# Memastikan direktori dataset ada
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_steam_review_data(app_id):
    """
    BATASAN PROPOSAL: Mengambil data ulasan positif (positive_reviews) 
    untuk keperluan mekanisme Tie-Breaker saat skor rekomendasi identik.
    """
    url = f"https://store.steampowered.com/appreviews/{app_id}?json=1"
    params = {'language': 'english', 'purchase_type': 'all'}
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if 'query_summary' in data:
                summary = data['query_summary']
                return {
                    'review_score_desc': summary.get('review_score_desc', 'No Rating'),
                    'positive_reviews': summary.get('total_positive', 0),
                    'total_reviews': summary.get('total_reviews', 0)
                }
    except Exception:
        pass
    return {'review_score_desc': 'Unknown', 'positive_reviews': 0, 'total_reviews': 0}


def get_app_ids_from_search(sort_filter, max_items):
    """
    Mengambil daftar App ID dari Steam Store Search.
    BATASAN PROPOSAL: Membatasi pencarian App ID agar tidak berlebihan.
    """
    app_ids = []
    start = 0
    print(f"[INFO] Mengumpulkan kandidat App ID ({sort_filter.upper()})...")
    while len(app_ids) < max_items:
        url = "https://store.steampowered.com/search/"
        params = {"category1": "998", "start": start, "count": 50}
        if sort_filter == "terbaru":
            params["sort_by"] = "Released_DESC"
        elif sort_filter == "terpopuler":
            params["filter"] = "topsellers"
            
        try:
            res = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if res.status_code == 429:
                print("[WARNING] Rate limit terdeteksi pada pencarian. Menunggu 60 detik...")
                time.sleep(60)
                continue
            if res.status_code == 200:
                found = re.findall(r'data-ds-appid=\"([^\"]+)\"', res.text)
                if not found:
                    break
                for attr in found:
                    for fid in attr.split(','):
                        if fid not in app_ids and len(app_ids) < max_items:
                            app_ids.append(fid)
                start += 50
                # BATASAN PROPOSAL: Delay untuk menghindari rate-limit API Steam
                time.sleep(1.5)
            else:
                break
        except Exception:
            break
    return app_ids


def get_app_details(app_id):
    """
    BATASAN PROPOSAL (PURPOSIVE SAMPLING):
    Hanya mengambil dan menyimpan game yang memiliki atribut teks lengkap ('detailed_description', 'genres').
    Game tanpa deskripsi detail atau genre akan dieliminasi dari dataset.
    """
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&l=english&cc=us"
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
        if res.status_code == 429:
            return {'status': 'rate_limited'}
            
        if res.status_code == 200:
            data = res.json()
            if data and data.get(str(app_id)) and data[str(app_id)]['success']:
                info = data[str(app_id)]['data']
                
                # Hanya simpan tipe 'game'
                if info.get('type') != 'game':
                    return {'status': 'not_game'}
                
                # --- PURPOSIVE SAMPLING CHECK ---
                detailed_desc = info.get('detailed_description', '').strip()
                genres_list = info.get('genres', [])
                genres_str = ";".join([g['description'] for g in genres_list]) if genres_list else ""
                
                # Seleksi ketat: Harus memiliki atribut teks lengkap (detailed_description & genres)
                if not detailed_desc or not genres_str:
                    return {'status': 'incomplete_data'}
                
                # Ambil data ulasan untuk Tie-Breaker
                review_data = get_steam_review_data(app_id)
                
                # BATASAN PROPOSAL: Delay antar HTTP Request agar tidak terkena rate-limit
                time.sleep(1.0)
                
                return {
                    'status': 'success',
                    'data': {
                        'steam_appid': int(app_id),
                        'name': info.get('name', ''),
                        'price': info.get('price_overview', {}).get('final', 0) / 100,
                        'detailed_description': detailed_desc,
                        'short_description': info.get('short_description', ''),
                        'genres': genres_str,
                        'tags': ";".join([c['description'] for c in info.get('categories', [])]),
                        'header_image': info.get('header_image', ''),
                        'rating': review_data['review_score_desc'],
                        'positive_reviews': review_data['positive_reviews'], # Untuk Tie-Breaker
                        'total_reviews': review_data['total_reviews']
                    }
                }
    except Exception:
        pass
    return {'status': 'error'}


# ==============================================================================
# MAIN SCRAPING LOGIC
# ==============================================================================
if __name__ == '__main__':
    print("[INFO] Memulai Smart Scraper Steam (Sesuai Proposal Skripsi)...")
    
    # 1. Check existing dataset
    existing_ids = set()
    total_existing = 0
    if os.path.exists(OUTPUT_FILE):
        try:
            df_existing = pd.read_csv(OUTPUT_FILE)
            if 'steam_appid' in df_existing.columns:
                existing_ids = set(df_existing['steam_appid'].astype(str).tolist())
                total_existing = len(existing_ids)
                print(f"[INFO] Data game yang sudah ada di dataset: {total_existing} game.")
        except Exception as e:
            print(f"[WARNING] Gagal membaca file existing: {e}")

    # BATASAN PROPOSAL: Cek apakah sudah mencapai batas 15.000 game
    if total_existing >= MAX_GAMES_LIMIT:
        print(f"[SUCCESS] Target batas Purposive Sampling ({MAX_GAMES_LIMIT} game) sudah terpenuhi di dataset.")
        sys.exit(0)

    # 2. Dapatkan kandidat App ID dari Steam Search
    target_fetch = MAX_GAMES_LIMIT - total_existing
    raw_ids = list(set(get_app_ids_from_search("terbaru", target_fetch) + get_app_ids_from_search("terpopuler", target_fetch)))
    new_ids = [aid for aid in raw_ids if str(aid) not in existing_ids]

    print(f"[INFO] Memproses {len(new_ids)} kandidat game baru (Target total dataset: {MAX_GAMES_LIMIT} game)...")

    scraped_data = []
    scraped_count = total_existing

    try:
        for i, app_id in enumerate(new_ids):
            # Batasan Purposive Sampling: Berhenti jika dataset sudah mencapai 15.000 game
            if scraped_count >= MAX_GAMES_LIMIT:
                print(f"[SUCCESS] Telah mencapai batas maksimal {MAX_GAMES_LIMIT} game sesuai Purposive Sampling.")
                break
                
            resp = get_app_details(app_id)
            
            if resp['status'] == 'success':
                scraped_data.append(resp['data'])
                scraped_count += 1
                print(f"[{scraped_count}/{MAX_GAMES_LIMIT}] Saved: {resp['data']['name']} (Positive Reviews: {resp['data']['positive_reviews']})")
                
                # Save berkala setiap 5 item
                if len(scraped_data) % 5 == 0:
                    df_temp = pd.DataFrame(scraped_data)
                    if os.path.exists(OUTPUT_FILE):
                        df_old = pd.read_csv(OUTPUT_FILE)
                        pd.concat([df_old, df_temp]).drop_duplicates(subset=['steam_appid']).to_csv(OUTPUT_FILE, index=False)
                    else:
                        df_temp.to_csv(OUTPUT_FILE, index=False)
                    scraped_data = []
                
                # BATASAN PROPOSAL: Delay tambahan setelah setiap penarikan sukses
                time.sleep(1.2)
                
            elif resp['status'] == 'rate_limited':
                print("[WARNING] Terkena rate limit dari Steam API. Menunggu 60 detik...")
                time.sleep(60)
            elif resp['status'] == 'incomplete_data':
                print(f"[SKIP] App ID {app_id} dilewati: data atribut teks tidak lengkap (Purposive Sampling).")
            elif resp['status'] == 'not_game':
                print(f"[SKIP] App ID {app_id} dilewati: bukan tipe game.")
                
    except KeyboardInterrupt:
        print("\n[STOP] Scraper dihentikan secara manual. Menyimpan data yang sudah ditarik...")
    finally:
        if scraped_data:
            df_final = pd.DataFrame(scraped_data)
            if os.path.exists(OUTPUT_FILE):
                df_old = pd.read_csv(OUTPUT_FILE)
                pd.concat([df_old, df_final]).drop_duplicates(subset=['steam_appid']).to_csv(OUTPUT_FILE, index=False)
            else:
                df_final.to_csv(OUTPUT_FILE, index=False)
        print(f"[SUCCESS] Progres tersimpan di: {OUTPUT_FILE}")