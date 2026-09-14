# LevelFind: Sistem Rekomendasi Game PC Berbasis Web

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20TF--IDF-orange.svg)](https://scikit-learn.org/)
[![Deployment](https://img.shields.io/badge/deployment-Vercel%20Serverless-black.svg)](https://vercel.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**LevelFind** adalah aplikasi sistem rekomendasi game PC berbasis web yang memanfaatkan metadata katalog Steam Store untuk memberikan rekomendasi game yang relevan dan personal. Menggunakan pendekatan **Content-Based Filtering** dengan pembobotan kata **TF-IDF** (_Term Frequency - Inverse Document Frequency_) dan metrik kedekatan **Cosine Similarity**, sistem ini mampu mengenali kesamaan genre, kategori, tag, hingga konteks semantik dari deskripsi game.

Proyek ini dirancang untuk menyelesaikan permasalahan _information overload_ bagi para gamer dalam mencari game alternatif yang memiliki karakteristik serupa (_lookalike_) dengan game favorit mereka, dilengkapi dengan fitur transparansi rekomendasi (**Explainable AI / XAI**).

---

## Daftar Isi

- [Fitur Utama](#fitur-utama)
- [Arsitektur & Tech Stack](#arsitektur--tech-stack)
- [Struktur Direktori Proyek](#struktur-direktori-proyek)
- [Instalasi & Panduan Menjalankan](#instalasi--panduan-menjalankan)
  - [Prasyarat Sistem](#prasyarat-sistem)
  - [Langkah Instalasi](#langkah-instalasi)
  - [Membuat Model & Artefak Data](#membuat-model--artefak-data)
  - [Menjalankan Aplikasi Secara Lokal](#menjalankan-aplikasi-secara-lokal)
- [Deployment Notes (Vercel Serverless)](#deployment-notes-vercel-serverless)
- [API Contracts & Endpoints](#api-contracts--endpoints)
- [Lisensi](#lisensi)

---

## Fitur Utama

1. **Content-Based Recommendation Engine**
   - Perhitungan kemiripan vektor konten secara instan menggunakan TF-IDF gabungan (_short description_, _detailed description_, _genres_, dan _user tags_).
   - **Tie-Breaker & Ranking Optimization**: Memprioritaskan game dengan ulasan positif lebih tinggi saat skor similaritas identik.
   - **Soft Penalty Sparse Corpus**: Menyesuaikan bobot game dengan deskripsi minim agar tidak mendominasi rekomendasi secara bias.

2. **Diversification & Novelty Filtering**
   - Menggunakan **Levenshtein Distance Ratio** untuk mendeteksi sekuel, edisi khusus (_Deluxe/GOTY_), atau judul yang hampir redundan (menghindari rekomendasi homogen seperti seri game berturut-turut).

3. **Explainable AI (XAI) / Transparansi Rekomendasi**
   - Menyediakan uraian naratif otomatis dalam dua bahasa (ID/EN) mengenai alasan sebuah game direkomendasikan berdasarkan fitur irisan utama (_Shared Genres & Dominant Tags_).

4. **Toleransi Typo & Autocomplete Cerdas**
   - Fitur _search autocomplete_ instan dan _fuzzy search fallback_ untuk menangani kesalahan pengetikan judul game oleh pengguna.

5. **Antarmuka Modern & Responsif (UI/UX)**
   - Mendukung **Bilingual Localization (Bahasa Indonesia & English)** secara real-time.
   - **Dark Mode & Light Mode** dengan transisi dinamis dan persistensi state `localStorage`.
   - Dynamic sorting & client-side filtering (berdasarkan _match percentage_, _rating_, dan _harga_).

6. **Production-Ready Security & Reliability**
   - Dilengkapi _Rate Limiter_ (`Flask-Limiter`), proteksi _CORS_, pembersihan input XSS (_Input Sanitization_), dan HTTP Security Headers (_nosniff, frame deny, referrer policy_).

---

## Arsitektur & Tech Stack

```
[ Pengguna / Browser ]
         │
         ▼  (HTTP GET / POST)
┌─────────────────────────────────────────────────────────────┐
│                       FLASK WEB SERVER                      │
│                                                             │
│  [ Security & Middleware ]                                  │
│  ├── RateLimiter (Flask-Limiter)                            │
│  ├── CORS (Flask-CORS)                                      │
│  └── Sanitize Input & Security Headers                      │
│                                                             │
│  [ Routing & Controller (app.py) ]                          │
│  ├── '/' (GET / POST: Render Jinja2 Template)               │
│  ├── '/api/search-suggestions' (Fuzzy Title Match)          │
│  └── '/api/recommend' (JSON Recommendation Endpoint)        │
│                                                             │
│  [ Recommendation & Similarity Engine ]                     │
│  ├── In-Memory Model Cache (@lru_cache)                     │
│  ├── Sparse Matrix Dot-Product (linear_kernel)              │
│  ├── Lexicographical Sorting (Lexsort: Score + Pos Reviews) │
│  ├── Diversification Filter (Levenshtein Distance)          │
│  └── Dynamic XAI Explainer (TF-IDF Feature Intersection)    │
│                                                             │
│  [ Data Artifacts (models/*.pkl) ]                          │
│  ├── game_data.pkl         (DataFrame Metadata Game)        │
│  ├── tfidf_matrix.pkl      (CSR Sparse TF-IDF Matrix)       │
│  ├── tfidf_vectorizer.pkl  (Fitted TfidfVectorizer)        │
│  └── indices.pkl           (Game Name to Positional Index)  │
└─────────────────────────────────────────────────────────────┘
         │
         ▼ (Rendered HTML / JSON)
[ Frontend: HTML5 + Jinja2 + Vanilla CSS + jQuery UI Autocomplete ]
```

### Rincian Teknologi:

- **Backend:** Python 3.9+, Flask
- **Machine Learning & NLP:** Scikit-Learn (`TfidfVectorizer`, `linear_kernel`), NumPy, SciPy (Sparse CSR Matrix)
- **String Processing:** `python-Levenshtein`
- **Security & Performance:** `Flask-Limiter`, `Flask-CORS`, Python `functools.lru_cache`
- **Frontend:** Jinja2 Template, HTML5, CSS3 kustom (Glassmorphism & CSS Variables), JavaScript (ES6+), jQuery
- **Deployment Platform:** Vercel Serverless Functions (`@vercel/python`)

---

## Struktur Direktori Proyek

```text
proyek_TA/
├── api/
│   └── index.py             # Entrypoint serverless function untuk deployment Vercel
├── dataset/
│   └── processed/           # Direktori penyimpanan dataset CSV hasil ekstraksi/scraping
├── models/
│   ├── game_data.pkl        # Serialisasi DataFrame game yang telah dibersihkan
│   ├── tfidf_matrix.pkl     # Matriks sparse CSR hasil ekstraksi fitur TF-IDF
│   ├── tfidf_vectorizer.pkl # Objek TF-IDF Vectorizer Scikit-Learn yang telah difit
│   └── indices.pkl          # Pemetaan judul game (lowercase) ke indeks baris DataFrame
├── static/
│   ├── css/
│   │   └── style.css        # Lembar gaya utama (tema gelap/terang, layout responsif)
│   └── js/
│       └── app.js           # Logika interaktif frontend, kamus i18n (ID/EN), event handlers
├── templates/
│   └── index.html           # Template utama Jinja2 untuk antarmuka pengguna
├── app.py                   # Aplikasi web utama Flask, API endpoint, & algoritma inferensi
├── build_model.py           # Skrip pipeline pembersihan data, vektorisasi, dan serialisasi model
├── smart_scraper.py         # Skrip otomatisasi penarikan metadata dan ulasan Steam Store API
├── requirements.txt         # Daftar pustaka dependensi Python
├── vercel.json              # Konfigurasi routing dan build Lambda untuk platform Vercel
├── .vercelignore            # Daftar berkas/folder yang dikecualikan dari bundle deployment Vercel
├── .gitignore               # Konfigurasi file yang diabaikan oleh Git
└── README.md                # Dokumentasi komprehensif repositori proyek
```

### Deskripsi Berkas & Modul Kunci:

- **`app.py`**: Inti aplikasi backend Flask. Bertanggung jawab memuat artefak model ke RAM secara in-memory, menangani validasi input pengguna, routing web/API, eksekusi Cosine Similarity dengan optimasi tie-breaker, hingga penyusunan teks XAI.
- **`build_model.py`**: Skrip _offline pipeline_ yang membaca dataset CSV mentah, membersihkan HTML/URL, menyaring _platform feature tags_, mengekstraksi representasi numerik menggunakan `TfidfVectorizer`, dan mengekspor artefak `.pkl` ke folder `models/`.
- **`api/index.py`**: Menjembatani aplikasi Flask (`app`) agar dapat dieksekusi sebagai WSGI Serverless Function di lingkungan cloud Vercel.
- **`models/game_data.pkl`**: Berkas artefak biner terkompresi berisi metadata esensial game (judul, genre, tags, ulasan, harga, gambar) yang dioptimasi untuk konsumsi memori rendah.
- **`templates/index.html`**: Antarmuka visual web yang dirender di sisi server (_SSR_) menggunakan Jinja2 dengan dukungan komponen interaktif modern.
- **`static/`**: Menyimpan aset statis pendukung, mencakup stylesheet bertema modern dan file JavaScript untuk interaksi sisi klien.

---

## Instalasi & Panduan Menjalankan

### Prasyarat Sistem

- **Python:** Versi 3.9, 3.10, atau 3.11
- **Pip:** Package installer bawaan Python
- **RAM Minimum:** 2 GB (direkomendasikan 4 GB untuk proses _build_ matriks TF-IDF)

### Langkah Instalasi

1. **Clone Repositori:**

   ```bash
   git clone https://github.com/Farhanrhm/Sistem-Rekomendasi-Game-PC-Berbasis-Web.git
   cd Sistem-Rekomendasi-Game-PC-Berbasis-Web
   ```

2. **Buat & Aktifkan Virtual Environment (Direkomendasikan):**
   - Di Windows (PowerShell):
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - Di macOS / Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependensi Proyek:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Membuat Model & Artefak Data

Sebelum menjalankan server web, pastikan berkas model `.pkl` sudah tersedia di dalam folder `models/`. Jika berkas belum ada atau Anda memperbarui dataset mentah di `dataset/processed/`:

```bash
python build_model.py
```

_Output yang diharapkan:_

```text
Membaca dataset...
Melakukan ekstraksi fitur teks gabungan dengan TfidfVectorizer (Scikit-Learn)...
Bentuk Matriks TF-IDF Sparse: (24119, ...)
Menyimpan file model .pkl ke folder 'models'...
[OK] --- PROSES PEMODELAN IN-MEMORY (.PKL) SELESAI ---
```

### Menjalankan Aplikasi Secara Lokal

Jalankan server pengembangan Flask:

```bash
python app.py
```

Buka web browser dan akses alamat:

```text
http://127.0.0.1:5000/
```

---

## Deployment Notes (Vercel Serverless)

Aplikasi ini telah dirancang agar kompatibel dengan lingkungan **Vercel Serverless Functions**:

1. **Struktur Entrypoint (`api/index.py`):**
   Vercel mengenali folder `api/` sebagai serverless endpoint. Berkas `api/index.py` mengimpor instance Flask `app` dari `app.py`.
2. **Konfigurasi `vercel.json`:**

   ```json
   {
     "version": 2,
     "builds": [
       {
         "src": "api/index.py",
         "use": "@vercel/python",
         "config": {
           "maxLambdaSize": "225mb",
           "excludeFiles": "{build_model.py,merge_dataset.py,smart_scraper.py,update_reviews_db.py,dataset/**,models/*_old*,*.csv,*.ipynb}"
         }
       }
     ],
     "routes": [{ "src": "/(.*)", "dest": "api/index.py" }]
   }
   ```

3. **Optimasi Ukuran Artefak (`.vercelignore`):**
   Batas ukuran bundle fungsi serverless Vercel (AWS Lambda) adalah 250 MB (unzipped). Oleh karena itu:
   - File mentah CSV, skrip scraping (`smart_scraper.py`), dan skrip training (`build_model.py`) **dikecualikan** melalui `.vercelignore`.
   - Hanya artefak kompresi inferensi di `models/` yang disertakan saat proses deployment.
   - Matriks TF-IDF disimpan dalam format Scipy Sparse Matrix (`csr_matrix`) untuk efisiensi konsumsi RAM dan disk.

---

## API Contracts & Endpoints

Selain melayani antarmuka web melalui server-side rendering, sistem ini menyediakan endpoint RESTful internal:

| Method         | Route                     | Parameter / Request Body                                   | Deskripsi                                                                           |
| -------------- | ------------------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `GET` / `POST` | `/`                       | Query `q` atau Form `game_title` (opsional: `top_n`)       | Merender halaman beranda atau hasil rekomendasi lengkap (HTML/Jinja2).              |
| `GET`          | `/api/search-suggestions` | Query `term` atau `q` (string)                             | Mengembalikan daftar judul game yang cocok untuk fitur autocomplete frontend.       |
| `GET` / `POST` | `/api/recommend`          | Query `q` / `game_title` atau JSON `{"game_title": "..."}` | Mengembalikan data rekomendasi game beserta penjelasan XAI dalam format JSON murni. |
| `GET`          | `/robots.txt`             | -                                                          | Menyediakan petunjuk perayapan untuk search engine bot.                             |
| `GET`          | `/sitemap.xml`            | -                                                          | Menyediakan peta situs XML untuk pengindeksan mesin pencari.                        |

### Rincian Kontrak Payload `/api/recommend`

#### Request Contoh (GET / POST)

```http
POST /api/recommend HTTP/1.1
Content-Type: application/json

{
  "game_title": "Elden Ring"
}
```

#### Response Sukses (200 OK)

```json
{
  "status": "success",
  "data": {
    "target_game": {
      "steam_appid": 1245620,
      "name": "ELDEN RING",
      "price": 59.99,
      "genres": "Action;RPG",
      "tags": "Dark Fantasy;Souls-like;Difficult;Open World",
      "header_image": "https://shared.cloudflare.steamstatic.com/store_item_assets/steam/apps/1245620/header.jpg",
      "short_description": "THE NEW FANTASY ACTION RPG...",
      "rating": "Very Positive",
      "rating_score": 92.0,
      "positive_reviews": 650000,
      "total_reviews": 700000,
      "similarity_score": 100.0,
      "is_sparse_corpus": false
    },
    "recommendations": [
      {
        "steam_appid": 374320,
        "name": "DARK SOULS™ III",
        "price": 59.99,
        "genres": "Action;RPG",
        "tags": "Souls-like;Dark Fantasy;Difficult",
        "header_image": "https://shared.cloudflare.steamstatic.com/store_item_assets/steam/apps/374320/header.jpg",
        "rating": "Very Positive",
        "rating_score": 94.0,
        "similarity_score": 0.84,
        "similarity_percentage": 84.2,
        "top_features": {
          "genres": [{ "name": "Action", "score": 0.31, "pct": 45 }],
          "tags": [{ "name": "Dark Fantasy", "score": 0.38, "pct": 55 }]
        },
        "xai_text_id": "Game ini direkomendasikan karena memiliki kesamaan Genre Action serta Tag Dark Fantasy dengan game pilihan Anda.",
        "xai_text_en": "This game is recommended because it shares the Action Genre and Dark Fantasy Tags with your selected game."
      }
    ],
    "corrected_from": null
  }
}
```

#### Response Error (404 Not Found)

```json
{
  "status": "error",
  "message": "Game 'Unknown Title' tidak ditemukan dalam sistem kami."
}
```

---

## Lisensi

Proyek ini didistribusikan di bawah lisensi **MIT License**. Lihat berkas [LICENSE](LICENSE) untuk informasi selengkapnya.

---

_Dikembangkan oleh [Farhan](https://github.com/Farhanrhm) sebagai Proyek Tugas Akhir Sistem Rekomendasi Game PC Berbasis Web._
