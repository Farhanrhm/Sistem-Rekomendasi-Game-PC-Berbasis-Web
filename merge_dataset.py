import pandas as pd
import os

# --- KONFIGURASI FILE ---
# Sesuaikan path ini dengan lokasi file CSV kamu
FILE_LAMA = 'dataset/processed/steam_new_and_fav.csv' 
FILE_BARU = 'dataset/processed/dataset_final_terbaru.csv' 
FILE_HASIL = 'dataset/processed/dataset_final_terbaru.csv'

def merge_datasets():
    # 1. Pengecekan ketersediaan file
    if not os.path.exists(FILE_LAMA):
        print(f"❌ Error: File lama '{FILE_LAMA}' tidak ditemukan.")
        return
    if not os.path.exists(FILE_BARU):
        print(f"❌ Error: File baru '{FILE_BARU}' tidak ditemukan.")
        return

    print("Memuat dataset...")
    # 2. Membaca kedua dataset ke dalam DataFrame Pandas
    df_lama = pd.read_csv(FILE_LAMA)
    df_baru = pd.read_csv(FILE_BARU)

    print(f"📊 Jumlah data lama: {len(df_lama)} baris")
    print(f"📊 Jumlah data baru: {len(df_baru)} baris")

    # 3. Menggabungkan kedua DataFrame (Baris baru ditumpuk di bawah baris lama)
    df_gabung = pd.concat([df_lama, df_baru], ignore_index=True)
    print(f"🔄 Jumlah total sebelum deduplikasi: {len(df_gabung)} baris")

    # 4. Menghapus duplikasi berdasarkan kolom 'steam_appid'
    # Parameter keep='last' berarti jika ada 2 game dengan ID sama, 
    # sistem akan menyimpan data yang paling baru (dari dataset baru)
    df_bersih = df_gabung.drop_duplicates(subset=['steam_appid'], keep='last')

    # 5. Menyimpan hasil gabungan yang sudah bersih
    df_bersih.to_csv(FILE_HASIL, index=False)
    
    jumlah_duplikat_dihapus = len(df_gabung) - len(df_bersih)
    print(f"🧹 Ditemukan dan dihapus {jumlah_duplikat_dihapus} data duplikat.")
    print(f"✅ Berhasil! Dataset akhir ({len(df_bersih)} baris) disimpan di: {FILE_HASIL}")

# Menjalankan fungsi
if __name__ == "__main__":
    merge_datasets()