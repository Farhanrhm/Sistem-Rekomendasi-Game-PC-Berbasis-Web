import requests
import pandas as pd
import time
import os
import re
import sys

# --- KONFIGURASI UTAMA ---
LIMIT_TERBARU = 10000
LIMIT_TERPOPULER = 10000

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}
OUTPUT_DIR = 'dataset/processed'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'steam_new_and_fav.csv')

# Pastikan direktori ada di awal
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_steam_review_data(app_id):
    url = f"https://store.steampowered.com/appreviews/{app_id}?json=1"
    params = {'language': 'english', 'purchase_type': 'all'}
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if 'query_summary' in data:
                return data['query_summary'].get('review_score_desc', 'No Rating')
    except: pass
    return "Unknown"

def get_app_ids_from_search(sort_filter, max_items):
    app_ids = []
    start = 0
    print(f"Mengumpulkan {max_items} App ID ({sort_filter.upper()})...")
    while len(app_ids) < max_items:
        url = "https://store.steampowered.com/search/"
        params = {"category1": "998", "start": start, "count": 50}
        if sort_filter == "terbaru": params["sort_by"] = "Released_DESC"
        elif sort_filter == "terpopuler": params["filter"] = "topsellers"
        try:
            res = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if res.status_code == 429: 
                time.sleep(120); continue
            if res.status_code == 200:
                found = re.findall(r'data-ds-appid=\"([^\"]+)\"', res.text)
                if not found: break
                for attr in found:
                    for fid in attr.split(','):
                        if fid not in app_ids and len(app_ids) < max_items: app_ids.append(fid)
                start += 50
                time.sleep(2)
            else: break
        except: break
    return app_ids

def get_app_details(app_id):
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&l=english&cc=us"
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
        if res.status_code == 429: return {'status': 'rate_limited'}
        if res.status_code == 200:
            data = res.json()
            if data and data.get(str(app_id)) and data[str(app_id)]['success']:
                info = data[str(app_id)]['data']
                if info.get('type') != 'game': return {'status': 'not_game'}
                return {
                    'status': 'success', 
                    'data': {
                        'steam_appid': int(app_id),
                        'name': info.get('name', ''),
                        'price': info.get('price_overview', {}).get('final', 0) / 100,
                        'rating': get_steam_review_data(app_id),
                        'genres': ";".join([g['description'] for g in info.get('genres', [])]),
                        'tags': ";".join([c['description'] for c in info.get('categories', [])]),
                        'header_image': info.get('header_image', '')
                    }
                }
    except: pass
    return {'status': 'error'}

# Main Logic
try:
    list_ids = list(set(get_app_ids_from_search("terbaru", LIMIT_TERBARU) + get_app_ids_from_search("terpopuler", LIMIT_TERPOPULER)))
    existing_ids = set()
    if os.path.exists(OUTPUT_FILE):
        existing_ids = set(pd.read_csv(OUTPUT_FILE, usecols=['steam_appid'])['steam_appid'].astype(str).tolist())
    
    new_ids = [aid for aid in list_ids if str(aid) not in existing_ids]
    print(f"Target: {len(new_ids)} game baru.")

    scraped_data = []
    for i, app_id in enumerate(new_ids):
        resp = get_app_details(app_id)
        if resp['status'] == 'success':
            scraped_data.append(resp['data'])
            print(f"[{i+1}/{len(new_ids)}] {resp['data']['name']}")
            
            # Autosave setiap 5 item agar file langsung muncul
            if (i + 1) % 5 == 0:
                df_temp = pd.DataFrame(scraped_data)
                if os.path.exists(OUTPUT_FILE):
                    df_old = pd.read_csv(OUTPUT_FILE)
                    pd.concat([df_old, df_temp]).drop_duplicates(subset=['steam_appid']).to_csv(OUTPUT_FILE, index=False)
                else:
                    df_temp.to_csv(OUTPUT_FILE, index=False)
                scraped_data = []
            time.sleep(1.5)
        elif resp['status'] == 'rate_limited':
            time.sleep(60)

except KeyboardInterrupt:
    print("\n🛑 Scraper dihentikan manual. Menyimpan data terakhir...")
finally:
    if scraped_data:
        df_final = pd.DataFrame(scraped_data)
        if os.path.exists(OUTPUT_FILE):
            df_old = pd.read_csv(OUTPUT_FILE)
            pd.concat([df_old, df_final]).drop_duplicates(subset=['steam_appid']).to_csv(OUTPUT_FILE, index=False)
        else:
            df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Progres tersimpan di: {OUTPUT_FILE}")