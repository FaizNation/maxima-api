# 🍊 Pomelo Leaf Disease Detection & Maxist AI Chatbot API
### *High-Performance Two-Step Verification AI & Multimodal Assistant for Citrus maxima Health Diagnosis*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-1.5_Flash-8E75B2.svg?logo=googlegemini&logoColor=white)](https://aistudio.google.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Swagger](https://img.shields.io/badge/Swagger-UI_Enabled-85EA2D.svg?logo=swagger&logoColor=black)](http://localhost:5000/apidocs)

---

## 📌 Gambaran Umum Proyek

**MAXIMA AI Microservice API** adalah layanan kecerdasan buatan berbasis Flask yang dirancang khusus untuk mendukung ekosistem pertanian modern pada komoditas jeruk bali (*Citrus maxima*). Layanan ini bertindak sebagai AI Microservice berkinerja tinggi yang dikonsumsi oleh backend utama (Express.js), menyediakan dua kapabilitas utama:

1. **Two-Step Verification Disease Classifier:** Sistem diagnosis bertahap yang menggabungkan model *Gatekeeper* (MobileNetV2 Sigmoid) untuk menyaring citra non-daun dan model *Expert* (VGG16 Softmax) untuk mengklasifikasikan 4 jenis kondisi daun, diperkaya oleh basis pengetahuan (*knowledge base*) rekomendasi penanganan.
2. **"Maxist" Multimodal AI Assistant:** Asisten konsultasi interaktif bertenaga **Google Gemini 1.5 Flash** yang mampu menjawab pertanyaan petani secara ramah dan sopan, menganalisis riwayat hasil pemindaian (`db_context`), serta meninjau foto daun secara multimodal via tautan gambar (`image_url`).

---

## 🤖 Fitur Chatbot Multimodal "Maxist" (Gemini 1.5 Flash)

**Maxist** hadir sebagai mitra cerdas bagi para petani jeruk bali dalam memahami kondisi kebun mereka secara komprehensif:

```
[ Express.js Backend / Frontend Client ]
                   │
                   ▼ (POST /api/v1/chat)
┌─────────────────────────────────────────────────────────────┐
│  Payload JSON:                                              │
│  - message    : Pertanyaan petani (wajib)                  │
│  - history    : Riwayat percakapan sesi (opsional)          │
│  - db_context : Data riwayat scan dari database (opsional)  │
│  - image_url  : Tautan foto daun jeruk bali (opsional)      │
└─────────────────────────────────────────────────────────────┘
                   │
         [ Apakah image_url ada? ]
         ├── YA ──► Unduh gambar ke buffer memori (In-Memory PIL RGB)
         │          Kirim [Image + Text Prompt + Konteks] ke Gemini 1.5 Flash
         │          (Analisis Citra Multimodal Langsung)
         │
         └── TIDAK ─► Susun riwayat percakapan (formatted_history)
                     Injeksi [Konteks Database] secara natural
                     Kirim percakapan ke Gemini 1.5 Flash Chat Session
                   │
                   ▼
       HTTP 200 OK: {"status": "success", "data": {"reply": "..."}}
```

### Karakteristik & Persona Maxist:
- **Ramah Petani:** Menggunakan bahasa Indonesia yang mudah dipahami, hangat, solutif, dan bebas jargon teknis yang membingungkan.
- **Konteks Database Natural:** Menjelaskan data riwayat scan tanpa menggunakan istilah teknis seperti "sistem" atau "database".
- **Multimodal Cepat & Akurat:** Menggunakan model `gemini-1.5-flash` dengan latensi rendah untuk inferensi teks maupun citra beresolusi tinggi.

---

## 🏛️ Arsitektur "Two-Step Verification"

Sistem klasifikasi penyakit daun mengeliminasi *false positives* dengan menyaring gambar acak (bukan daun jeruk bali) sebelum diproses ke model pakar:

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

---

## 📂 Struktur Proyek Modular

```text
maxima-api/
├── api/
│   ├── __init__.py               # Inisialisasi package API
│   └── routes.py                 # Endpoint GET /, POST /api/v1/predict, & POST /api/v1/chat
├── services/
│   ├── __init__.py               # Inisialisasi package services
│   ├── ai_service.py             # Preprocessing citra, loading Keras model, & pipeline 2-tahap
│   └── chat_service.py           # Layanan Chatbot Maxist multimodal (Gemini 1.5 Flash)
├── utils/
│   ├── __init__.py               # Inisialisasi package utils
│   └── knowledge_base.py         # Basis data edukasi penyakit dan penanganan
├── app.py                        # Entry point Flask, CORS, Swagger UI, & global error handling
├── requirements.txt              # Dependensi pustaka Python
├── Dockerfile                    # Containerization berbasis python:3.10-slim & Gunicorn
├── docker-compose.yml            # Orkestrasi container Docker Compose
├── .dockerignore                 # Pengecualian berkas build image Docker
├── .gitignore                    # Pengecualian berkas version control Git
├── .env.example                  # Template variabel lingkungan
├── model_satpam_pomelo.h5        # Model Gatekeeper (MobileNetV2 Sigmoid)
├── pomelo_disease_model.h5       # Model Expert (VGG16 Softmax 4 Kelas)
├── pomelo_labels.json            # Daftar label kelas klasifikasi
└── README.md                     # Dokumentasi resmi proyek
```

---

## ⚙️ Konfigurasi Environment Variables

Salin berkas template `.env.example` menjadi `.env`:
```bash
cp .env.example .env
```

Isi variabel konfigurasi berikut:
```env
# Port & Host server Flask
PORT=5000
HOST=0.0.0.0

# API Key Google Gemini untuk Asisten Maxist (Wajib untuk fitur chat)
# Dapatkan gratis di: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=AIzaSyYourGeminiApiKeyHere
```

---

## 🚀 Panduan Instalasi & Menjalankan Server

### Prasyarat Sistem
- Python 3.10+
- Git
- Docker & Docker Compose (opsional)

---

### Opsi 1: Menjalankan Secara Lokal (Virtualenv)

1. **Masuk ke Direktori Proyek:**
   ```bash
   cd maxima-api
   ```

2. **Buat & Aktifkan Virtual Environment:**
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

4. **Jalankan Aplikasi:**
   - **Mode Development:**
     ```bash
     python app.py
     ```
   - **Mode Production (Gunicorn):**
     ```bash
     gunicorn --bind 0.0.0.0:5000 --workers 2 --threads 4 --timeout 120 app:app
     ```

5. Server aktif di: `http://localhost:5000`

---

### Opsi 2: Menjalankan Menggunakan Docker & Docker Compose

1. **Build dan Jalankan Container:**
   ```bash
   docker-compose up --build -d
   ```

2. **Cek Status Container:**
   ```bash
   docker-compose ps
   ```

3. **Melihat Log Aplikasi:**
   ```bash
   docker-compose logs -f pomelo-api
   ```

4. **Hentikan Container:**
   ```bash
   docker-compose down
   ```

---

## 📖 Dokumentasi Interaktif Swagger UI

Akses dokumentasi interaktif OpenAPI / Swagger UI langsung melalui browser:

👉 **URL Swagger UI:** [http://localhost:5000/apidocs](http://localhost:5000/apidocs)

Fitur Swagger UI:
- Menguji endpoint klasifikasi daun dengan mengunggah gambar langsung (*multipart/form-data*).
- Menguji endpoint percakapan asisten Maxist dengan payload JSON interaktif (*Try it out*).

---

## 📡 Dokumentasi Endpoint REST API

### 1. Health Check
- **Method:** `GET`
- **Path:** `/`
- **Response (`200 OK`):**
```json
{
  "status": "success",
  "message": "Pomelo Disease Detection API is running healthy.",
  "version": "1.1.0",
  "architecture": "Two-Step Verification (Gatekeeper MobileNetV2 + Expert VGG16) & Maxist Chatbot (Gemini 1.5 Flash)",
  "classes": [
    "Pomelo_Cephaleuros_virescens",
    "Pomelo_Healthy",
    "Pomelo_Leaf_Miner",
    "Pomelo_Orange_Mold"
  ]
}
```

---

### 2. Chatbot Asisten Maxist (Multimodal)
- **Method:** `POST`
- **Path:** `/api/v1/chat`
- **Content-Type:** `application/json`

#### Schema Payload Request:
| Field | Tipe | Status | Deskripsi |
|---|---|:---:|---|
| `message` | `string` | **Wajib** | Pertanyaan atau pesan dari pengguna/petani. |
| `history` | `array` | Opsional | Riwayat percakapan sesi sebelumnya `[{"role": "user"|"model", "parts": [...]}]`. |
| `db_context` | `string` | Opsional | Ringkasan riwayat scan dari database Express.js. |
| `image_url` | `string` | Opsional | URL gambar daun jeruk bali jika ingin dibahas bersama asisten. |

#### Contoh Request cURL:
```bash
curl -X POST "http://localhost:5000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Bagaimana cara penanganan bercak ganggang pada daun jeruk bali saya?",
    "db_context": "Hasil scan terakhir: Terindikasi Bercak Ganggang (Cephaleuros virescens) dengan keyakinan 98.45%.",
    "image_url": "https://example.com/uploads/scan_daun_pomelo.jpg",
    "history": [
      {
        "role": "user",
        "parts": ["Halo Maxist, saya petani jeruk bali."]
      },
      {
        "role": "model",
        "parts": ["Halo Bapak/Ibu Petani! Senang bertemu Anda. Ada yang bisa Maxist bantu seputar tanaman jeruk bali Anda?"]
      }
    ]
  }'
```

#### Contoh Request Python:
```python
import requests

url = "http://localhost:5000/api/v1/chat"
payload = {
    "message": "Tolong beri rekomendasi pupuk untuk menjaga daun jeruk bali tetap sehat.",
    "db_context": "Pohon berusia 3 tahun, daun tergolong sehat."
}

response = requests.post(url, json=payload)
print(response.json())
```

#### Contoh Response Sukses (`200 OK`):
```json
{
  "status": "success",
  "data": {
    "reply": "Halo Bapak/Ibu Petani! Berdasarkan kondisi pohon jeruk bali Anda yang berusia 3 tahun dan dalam kondisi daun yang prima, kami menyarankan pemberian pupuk NPK seimbang (15-15-15) secara berkala setiap 3-4 bulan sekali. Selain itu, berikan pupuk kandang matang di sekeliling tajuk tanaman untuk menjaga kesuburan mikroorganisme tanah..."
  }
}
```

#### Contoh Response Validasi Gagal (`400 Bad Request`):
```json
{
  "status": "fail",
  "message": "Parameter 'message' wajib diisi dan tidak boleh kosong."
}
```

#### Contoh Response API Key Belum Disetel (`500 Internal Server Error`):
```json
{
  "status": "error",
  "message": "GEMINI_API_KEY belum dikonfigurasi di environment variables. Harap tambahkan GEMINI_API_KEY pada file .env atau variabel lingkungan sistem."
}
```

---

### 3. Klasifikasi Penyakit Daun (Two-Step Verification)
- **Method:** `POST`
- **Path:** `/api/v1/predict`
- **Content-Type:** `multipart/form-data`
- **Body Parameter:**
  - `file` (*binary/file*, wajib): Citra daun jeruk bali (`.jpg`, `.jpeg`, `.png`, `.webp`).

#### Contoh Request cURL:
```bash
curl -X POST "http://localhost:5000/api/v1/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/daun_jeruk_bali.jpg"
```

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
      "deskripsi": "Penyakit bercak ganggang disebabkan oleh alga parasit Cephaleuros virescens...",
      "penanganan": [
        "Pangkas daun dan ranting yang terinfeksi berat lalu bakar atau musnahkan.",
        "Semprotkan fungisida berbahan aktif tembaga secara merata.",
        "Lakukan pemangkasan tajuk untuk memperbaiki sirkulasi udara dan intensitas cahaya matahari."
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

#### Contoh Response Ditolak Satpam / Gatekeeper (`400 Bad Request`):
```json
{
  "status": "fail",
  "message": "Gambar tidak valid. Objek bukan daun Jeruk Bali.",
  "satpam_score_persen": 12.4
}
```

---

## 🤝 Panduan Kontribusi

1. **Fork Repositori ini**.
2. **Buat Feature Branch:**
   ```bash
   git checkout -b feature/nama-fitur-baru
   ```
3. **Commit Perubahan:**
   ```bash
   git commit -m "feat: Menambahkan fitur rekomendasi cuaca perkebunan"
   ```
4. **Push ke Branch Anda:**
   ```bash
   git push origin feature/nama-fitur-baru
   ```
5. **Buka Pull Request (PR)** dengan deskripsi pengujian yang jelas.

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).
