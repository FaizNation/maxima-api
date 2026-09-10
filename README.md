# 🍊 Pomelo Leaf Disease Detection API
### *High-Performance Two-Step Verification AI System for Citrus maxima Health Diagnosis*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Swagger](https://img.shields.io/badge/Swagger-UI_Enabled-85EA2D.svg?logo=swagger&logoColor=black)](http://localhost:5000/apidocs)

---

## 📌 Gambaran Umum Proyek

**Pomelo Leaf Disease Detection API** adalah layanan RESTful API berbasis microservice yang dirancang khusus untuk mendeteksi dan mengklasifikasikan penyakit pada daun jeruk bali (*Citrus maxima*). Sistem ini mengusung arsitektur inovatif **"Two-Step Verification" (Gatekeeper & Expert Model)** guna memastikan tingkat akurasi tinggi serta mencegah terjadinya *false positive* dari citra non-target.

Layanan ini dilengkapi dengan basis pengetahuan (*knowledge base*) komprehensif berbahasa Indonesia yang secara otomatis memberikan nama umum penyakit, tingkat bahaya, deskripsi klinis, dan langkah-langkah penanganan serta mitigasi bagi petani dan praktisi pertanian.

---

## 🏛️ Arsitektur "Two-Step Verification"

Dalam implementasi *computer vision* dunia nyata, pengguna sering kali mengunggah gambar latar belakang acak, tangan, tanah, atau daun tanaman lain. Untuk mengatasi kelemahan model klasifikasi tunggal yang memaksakan prediksi pada objek acak, kami menerapkan arsitektur dua tahap:

```
[ User Uploads Image ]
          │
          ▼
┌───────────────────────────────────────────────┐
│              TAHAP PREPROCESSING              │
│  - Pillow RGB conversion (no alpha channel)   │
│  - Bilinear Resize ke 224x224 piksel          │
│  - Normalisasi numpy float32                  │
│  - Dimensi batch expansion (1, 224, 224, 3)   │
└───────────────────────────────────────────────┘
          │
          ▼
┌───────────────────────────────────────────────┐
│     LANGKAH 1: MODEL SATPAM (GATEKEEPER)      │
│  - Arsitektur : MobileNetV2                   │
│  - Aktivasi   : Sigmoid (Output biner: 0 s/d 1│
│  - File       : model_satpam_pomelo.h5        │
└───────────────────────────────────────────────┘
          │
     [ Skor >= 0.5 ? ]
     ├── TIDAK (< 0.5) ──► HTTP 400 Bad Request
     │                     "Gambar tidak valid. Objek bukan daun Jeruk Bali."
     │
     └── YA (>= 0.5)
          │
          ▼
┌───────────────────────────────────────────────┐
│     LANGKAH 2: MODEL PAKAR (EXPERT MODEL)     │
│  - Arsitektur : VGG16 Transfer Learning       │
│  - Aktivasi   : Softmax (4 Kelas Penyakit)    │
│  - File       : pomelo_disease_model.h5       │
└───────────────────────────────────────────────┘
          │
          ▼
┌───────────────────────────────────────────────┐
│         ENRICHMENT DENGAN KNOWLEDGE BASE      │
│  - Mapping ID kelas ke utils/knowledge_base.py │
│  - Ambil Nama Ilmiah, Nama Umum, Bahaya,      │
│    Deskripsi, dan Rekomendasi Solusi/Tindakan │
└───────────────────────────────────────────────┘
          │
          ▼
    HTTP 200 OK (Structured JSON Response)
```

### Keunggulan Arsitektur Ini:
1. **Pencegahan Halusinasi Model**: Citra non-daun jeruk bali langsung ditolak di pintu gerbang (*gatekeeper*) tanpa membebani model pakar.
2. **Efisiensi Sumber Daya**: MobileNetV2 sangat ringan dan cepat mengeksekusi penyaringan awal.
3. **Akurasi Diagnostik Tinggi**: Model pakar VGG16 hanya berfokus pada diferensiasi penyakit daun jeruk bali yang valid.

---

## 📂 Struktur Proyek Modular

Struktur direktori disusun secara rapi dan modular sesuai prinsip *Clean Architecture* dan *Separation of Concerns*:

```text
maxima-api/
├── api/
│   ├── __init__.py               # Inisialisasi package API
│   └── routes.py                 # Definisi routing, validasi input, dan Swagger docstrings
├── services/
│   ├── __init__.py               # Inisialisasi package services
│   └── ai_service.py             # Preprocessing citra, loading model, dan Two-Step pipeline
├── utils/
│   ├── __init__.py               # Inisialisasi package utils
│   └── knowledge_base.py         # Basis data edukasi penyakit dan penanganan
├── app.py                        # Entry point Flask, CORS, konfigurasi Swagger, dan error handlers
├── requirements.txt              # Daftar dependensi pustaka Python
├── Dockerfile                    # Konfigurasi container image Docker berbasis python:3.10-slim
├── docker-compose.yml            # Konfigurasi orkestrasi Docker Compose
├── .dockerignore                 # Berkas yang dikecualikan dari proses build image Docker
├── model_satpam_pomelo.h5        # Model Gatekeeper (MobileNetV2 Sigmoid)
├── pomelo_disease_model.h5       # Model Expert (VGG16 Softmax 4 Kelas)
├── pomelo_labels.json            # Daftar label kelas klasifikasi
└── README.md                     # Dokumentasi resmi proyek
```

---

## 🌿 Knowledge Base Penyakit Daun Jeruk Bali

| ID Kelas JSON | Nama Umum | Tingkat Bahaya | Deskripsi Utama | Tindakan Utama |
|---|---|:---:|---|---|
| `Pomelo_Cephaleuros_virescens` | Bercak Ganggang | **Sedang** | Bercak menonjol hijau kelabu hingga jingga akibat alga parasit *Cephaleuros virescens*. | Pangkas daun sakit, semprot fungisida tembaga, perbaiki aerasi tajuk. |
| `Pomelo_Leaf_Miner` | Hama Pengorok Daun | **Tinggi** | Liang berliku keperakan pada daun muda akibat larva *Phyllocnistis citrella*. | Petik dan bakar daun bergejala, semprot minyak mimba (*neem oil*), pasang perangkap kuning. |
| `Pomelo_Orange_Mold` | Jamur Kapang Oranye | **Tinggi** | Koloni jamur oranye tebal memicu klorosis hebat dan daun rontok prematur. | Isolasi tanaman, sanitasi gulma, semprot fungisida sistemik spektrum luas. |
| `Pomelo_Healthy` | Daun Sehat | **Aman** | Daun segar, turgor baik, tidak ada gejala infeksi hama maupun jamur. | Lanjutkan pemupukan berimbang rutin, pertahankan drainase tanah prima. |

---

## 🚀 Panduan Instalasi & Menjalankan Server

### Prasyarat Sistem
- Python 3.10+
- Pip & Virtualenv
- Git (opsional)
- Docker & Docker Compose (jika menjalankan via container)

---

### Opsi 1: Menjalankan Secara Lokal (Virtualenv)

1. **Clone atau Masuk ke Direktori Proyek:**
   ```bash
   cd maxima-api
   ```

2. **Buat dan Aktifkan Virtual Environment:**
   - **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

3. **Install Dependensi:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Jalankan Server:**
   - **Mode Development (Flask):**
     ```bash
     python app.py
     ```
   - **Mode Production (Gunicorn - Linux / WSL / Container):**
     ```bash
     gunicorn --bind 0.0.0.0:5000 --workers 2 --threads 4 --timeout 120 app:app
     ```

5. Server aktif di: `http://localhost:5000`

---

### Opsi 2: Menjalankan Menggunakan Docker & Docker Compose

Proyek ini telah dikonfigurasi penuh untuk berjalan di dalam container Docker yang ringan berbasis `python:3.10-slim`.

1. **Build dan Jalankan Container:**
   ```bash
   docker-compose up --build -d
   ```

2. **Mengecek Status Container:**
   ```bash
   docker-compose ps
   ```

3. **Melihat Log Aplikasi:**
   ```bash
   docker-compose logs -f pomelo-api
   ```

4. **Menghentikan Container:**
   ```bash
   docker-compose down
   ```

---

## 📖 Dokumentasi Interaktif Swagger UI

Aplikasi dilengkapi dengan dokumentasi interaktif OpenAPI / Swagger UI yang dapat diakses langsung melalui browser:

👉 **URL Swagger UI:** [http://localhost:5000/apidocs](http://localhost:5000/apidocs)

Melalui antarmuka Swagger, Anda dapat:
- Melihat spesifikasi lengkap schema request dan response.
- Mengunggah file gambar daun jeruk bali secara langsung (*Try it out*) untuk menguji inferensi tanpa perlu tools eksternal.

---

## 📡 Dokumentasi Endpoint REST API

### 1. Health Check & Informasi Sistem
- **Method:** `GET`
- **Path:** `/`
- **Deskripsi:** Memeriksa status kesehatan server dan daftar kelas model AI aktif.

**Contoh Response (`200 OK`):**
```json
{
  "status": "success",
  "message": "Pomelo Disease Detection API is running healthy.",
  "version": "1.0.0",
  "architecture": "Two-Step Verification (Gatekeeper MobileNetV2 + Expert VGG16)",
  "classes": [
    "Pomelo_Cephaleuros_virescens",
    "Pomelo_Healthy",
    "Pomelo_Leaf_Miner",
    "Pomelo_Orange_Mold"
  ]
}
```

---

### 2. Klasifikasi Penyakit Daun (Two-Step Verification)
- **Method:** `POST`
- **Path:** `/api/v1/predict`
- **Content-Type:** `multipart/form-data`
- **Body Parameter:**
  - `file` (*binary/file*, wajib): File citra daun jeruk bali (`.jpg`, `.jpeg`, `.png`, `.webp`).

#### Contoh Pengujian via cURL:
```bash
curl -X POST "http://localhost:5000/api/v1/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/daun_jeruk_bali.jpg"
```

#### Contoh Pengujian via Python Requests:
```python
import requests

url = "http://localhost:5000/api/v1/predict"
with open("daun_jeruk.jpg", "rb") as img:
    files = {"file": ("daun_jeruk.jpg", img, "image/jpeg")}
    response = requests.post(url, files=files)
    print(response.json())
```

---

#### Contoh Response Sukses (`200 OK`):
```json
{
  "status": "success",
  "message": "Klasifikasi penyakit daun jeruk bali berhasil.",
  "data": {
    "id_kelas": "Pomelo_Cephaleuros_virescens",
    "tingkat_keyakinan_persen": 98.45,
    "detail_penyakit": {
      "nama_ilmiah": "Cephaleuros virescens (Algal Spot)",
      "nama_umum": "Bercak Ganggang",
      "bahaya": "Sedang",
      "deskripsi": "Penyakit bercak ganggang disebabkan oleh alga parasit Cephaleuros virescens. Gejala ditandai dengan munculnya bercak melingkar yang agak menonjol seperti beludru berwarna hijau kelabu hingga jingga/karat pada permukaan atas daun jeruk bali.",
      "penanganan": [
        "Pangkas daun dan ranting yang terinfeksi berat lalu bakar atau musnahkan jauh dari kebun.",
        "Semprotkan fungisida berbahan aktif tembaga (copper-based fungicide seperti tembaga oksiklorida) secara merata.",
        "Lakukan pemangkasan cabang secara berkala untuk meningkatkan sirkulasi udara dan penetrasi sinar matahari ke dalam tajuk tanaman.",
        "Tingkatkan vigor tanaman melalui pemupukan seimbang, terutama kalium dan unsur mikro."
      ]
    },
    "verifikasi_satpam": {
      "lulus": true,
      "skor_keyakinan_daun_persen": 99.85
    },
    "probabilitas_semua_kelas": {
      "Pomelo_Cephaleuros_virescens": 98.45,
      "Pomelo_Healthy": 0.32,
      "Pomelo_Leaf_Miner": 0.81,
      "Pomelo_Orange_Mold": 0.42
    }
  }
}
```

---

#### Contoh Response Ditolak Satpam / Gatekeeper (`400 Bad Request`):
```json
{
  "status": "fail",
  "message": "Gambar tidak valid. Objek bukan daun Jeruk Bali.",
  "satpam_score_persen": 12.4
}
```

---

#### Contoh Response Kesalahan Input Pengguna (`400 Bad Request`):
```json
{
  "status": "fail",
  "message": "Key 'file' tidak ditemukan dalam multipart form-data. Pastikan key input bernilai 'file'."
}
```

---

#### Contoh Response Kesalahan Server (`500 Internal Server Error`):
```json
{
  "status": "error",
  "message": "Terjadi kesalahan internal server saat memproses gambar."
}
```

---

## 🤝 Panduan Kontribusi

Kami menyambut gembira kontribusi dari komunitas peneliti AI, developer, dan praktisi pertanian! Untuk berkontribusi:

1. **Fork Repositori ini**.
2. **Buat Feature Branch:**
   ```bash
   git checkout -b feature/FiturKerenBaru
   ```
3. **Lakukan Perubahan & Tulis Test Case:**
   - Pastikan kode mengikuti standar PEP 8.
   - Tambahkan dokumentasi jika mengubah/menambahkan fungsi.
4. **Commit Perubahan:**
   ```bash
   git commit -m "feat: Menambahkan fitur deteksi segmentasi daun"
   ```
5. **Push ke Branch Anda:**
   ```bash
   git push origin feature/FiturKerenBaru
   ```
6. **Buka Pull Request (PR):** Berikan deskripsi yang jelas mengenai alasan dan pengujian fitur yang telah dibuat.

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE). Bebas digunakan, dimodifikasi, dan didistribusikan untuk kepentingan edukasi, penelitian, maupun komersial.
