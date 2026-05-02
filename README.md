# LevelFind: Sistem Rekomendasi Game PC Berbasis Web

**LevelFind** adalah aplikasi web yang dirancang untuk membantu pengguna menemukan rekomendasi game PC yang relevan berdasarkan kemiripan konten (_Content-Based Filtering_). Proyek ini dikembangkan sebagai bagian dari Tugas Akhir/Skripsi dengan studi kasus dataset dari **Steam Store**.

---

## Fitur Utama

- **Data Scraping**: Menggunakan Steam Store API untuk mendapatkan informasi game yang komprehensif (Genre, Tags, Rating, Deskripsi).
- **Recommendation Engine**: Menggunakan pemrosesan bahasa alami (NLP) dengan metode TF-IDF dan perhitungan jarak Cosine Similarity.
- **Transparansi Sistem**: Fitur penjelasan yang membantu pengguna memahami mengapa sebuah game direkomendasikan.
- **Modern UI/UX**: Antarmuka responsif dengan fitur _Dark/Light Mode_, _Search Autocomplete_, dan _Skeleton Loading_.
- **Performance Optimized**: Implementasi _Client-side Sorting_ dan _Server-side Caching_ (LRU Cache) untuk pengalaman pengguna yang cepat.

---

## Stack Teknologi

| Komponen             | Teknologi                                                |
| -------------------- | -------------------------------------------------------- |
| **Backend**          | Python (Flask Framework)                                 |
| **Similarity Engine** | Scikit-Learn (TF-IDF, Cosine Similarity), Pandas, NumPy  |
| **Frontend**         | HTML5, CSS3 (Modern Flexbox/Grid), JavaScript, jQuery UI |
| **Dataset**          | Steam Store API                                          |
| **Version Control**  | Git & GitHub                                             |

---

## Metodologi

Sistem ini bekerja dengan menganalisis metadata tekstual game (Judul, Genre, Tags, dan Deskripsi). Proses perhitungan kemiripan dilakukan melalui tahapan berikut:

1. **Text Preprocessing**: Membersihkan tag HTML, simbol, dan melakukan normalisasi teks.
2. **TF-IDF Vectorization**: Mengubah korpus teks menjadi representasi numerik (vektor) berdasarkan bobot kepentingan kata.
3. **Cosine Similarity**: Menghitung sudut kosinus antara dua vektor untuk menentukan tingkat kemiripan.

$$\text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$

---

## Dataset

Data diperoleh melalui proses penggabungan data menggunakan web scraping terkendali dari:

1. Steam Store API: Untuk deskripsi resmi, harga, dan gambar header.
