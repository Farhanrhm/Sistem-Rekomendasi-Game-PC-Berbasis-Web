import requests
import pandas as pd
import time
import os
import random

# --- KONFIGURASI ---
LIMIT = 100    
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
OUTPUT_DIR = 'dataset/processed'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'steam_merged_clean.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_app_details(app_id):
    try:
        url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&l=english&cc=us"
        res = requests.get(url, headers=HEADERS, timeout=10)
        if res.status_code != 200: return None
        data = res.json()
        if data and data.get(str(app_id)) and data[str(app_id)]['success']:
            info = data[str(app_id)]['data']
            if info.get('type') != 'game': return None
            return {
                'steam_appid': app_id,
                'name': info.get('name'),
                'price': info.get('price_overview', {}).get('final', 0) / 100,
                'detailed_description': info.get('detailed_description', ''),
                'genres': ";".join([g['description'] for g in info.get('genres', [])]),
                'header_image': info.get('header_image')
            }
    except: return None
    return None

# Daftar URL cadangan untuk mengambil List Game
urls_to_try = [
    "https://api.steampowered.com/ISteamApps/GetAppList/v2",
    "https://api.steampowered.com/ISteamApps/GetAppList/v0002",
    "https://community.steam-api.com/ISteamApps/GetAppList/v2"
]

all_apps = []
print("Mengambil daftar App ID dari Steam...")

for url in urls_to_try:
    try:
        print(f"Mencoba URL: {url}")
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            all_apps = response.json()['applist']['apps']
            print(f"Berhasil! Ditemukan {len(all_apps)} game.")
            break
        else:
            print(f"Gagal (Status {response.status_code})")
    except:
        continue

if not all_apps:
    print("❌ Semua API List Steam gagal diakses. Coba gunakan VPN atau ganti koneksi internet.")
    exit()

random.shuffle(all_apps)
scraped_data = []
count = 0

print(f"Mulai scraping {LIMIT} game...")
for app in all_apps:
    if count >= LIMIT: break
    details = get_app_details(app['appid'])
    if details and details['name']:
        scraped_data.append(details)
        count += 1
        print(f"[{count}/{LIMIT}] Berhasil: {details['name']}")
        time.sleep(1.5)

if scraped_data:
    pd.DataFrame(scraped_data).to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Selesai! Data disimpan di: {OUTPUT_FILE}")