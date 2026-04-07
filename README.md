# LevelFind: Sistem Rekomendasi Game PC Berbasis Web

**LevelFind** adalah aplikasi web berbasis AI yang dirancang untuk membantu pengguna menemukan rekomendasi game PC yang relevan berdasarkan kemiripan konten (*Content-Based Filtering*). Proyek ini dikembangkan sebagai bagian dari Tugas Akhir/Skripsi dengan studi kasus dataset dari **Steam Store**.

---

##  Fitur Utama
- **Hybrid Data Scraping**: Menggabungkan data dari SteamSpy API dan Steam Store API untuk mendapatkan informasi game yang komprehensif (Genre, Tags, Rating, Deskripsi).
- **AI Recommendation Engine**: Menggunakan pemrosesan bahasa alami (NLP) dengan metode TF-IDF dan perhitungan jarak Cosine Similarity.
- **Explainable AI (XAI)**: Fitur transparansi yang menjelaskan kepada pengguna mengapa sebuah game direkomendasikan.
- **Modern UI/UX**: Antarmuka responsif dengan fitur *Dark/Light Mode*, *Search Autocomplete*, dan *Skeleton Loading*.
- **Performance Optimized**: Implementasi *Client-side Sorting* dan *Server-side Caching* (LRU Cache) untuk pengalaman pengguna yang cepat.

---

##  Stack Teknologi
| Komponen | Teknologi |
| --- | --- |
| **Backend** | Python (Flask Framework) |
| **Machine Learning** | Scikit-Learn (TF-IDF, Cosine Similarity), Pandas, NumPy |
| **Frontend** | HTML5, CSS3 (Modern Flexbox/Grid), JavaScript, jQuery UI |
| **Dataset** | Steam Store API & SteamSpy API |
| **Version Control** | Git & GitHub |

---

##  Metodologi
Sistem ini bekerja dengan menganalisis metadata tekstual game (Judul, Genre, Tags, dan Deskripsi). Proses perhitungan kemiripan dilakukan melalui tahapan berikut:

1. **Text Preprocessing**: Membersihkan tag HTML, simbol, dan melakukan normalisasi teks.
2. **TF-IDF Vectorization**: Mengubah korpus teks menjadi representasi numerik (vektor) berdasarkan bobot kepentingan kata.
3. **Cosine Similarity**: Menghitung sudut kosinus antara dua vektor untuk menentukan tingkat kemiripan.

$$\text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$

---

##  Instalasi dan Penggunaan

### 1. Clone Repository
```bash
git clone [https://github.com/Farhanrhm/Sistem-Rekomendasi-Game-PC-Berbasis-Web.git](https://github.com/Farhanrhm/Sistem-Rekomendasi-Game-PC-Berbasis-Web.git)
cd Sistem-Rekomendasi-Game-PC-Berbasis-Web
```

### 2. Install Dependencies
Pastikan Anda memiliki Python 3.x terinstall.
```bash
pip install -r requirements.txt
```
### 3. Persiapkan Dataset dan Model
Jika Anda ingin memperbarui data, jalankan script secara berurutan:
```bash
python smart_scraper.py
python 2_text_preprocessing.py
python 3_build_model.py
```
### 4. Jalankan Aplikasi
```bash
python app.py
```
---
Akses aplikasi melalui browser di: http://127.0.0.1:5000

## Dataset
Data diperoleh secara hibrida menggunakan web scraping terkendali dari:
1. SteamSpy: Untuk metadata rating, owners, dan tags.
2. Steam Store API: Untuk deskripsi resmi, harga, dan gambar header.
