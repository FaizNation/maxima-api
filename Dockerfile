# ==============================================================================
# Dockerfile: Pomelo Leaf Disease Detection API
# Arsitektur: Two-Step Verification (MobileNetV2 Gatekeeper + VGG16 Expert)
# ==============================================================================

# Gunakan base image resmi Python 3.10 slim
FROM python:3.10-slim

# Label informasi metadata pembuat image
LABEL maintainer="AI Engineering Team"
LABEL description="RESTful API Klasifikasi Penyakit Daun Jeruk Bali dengan Two-Step Verification"

# Atur environment variable
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    TF_CPP_MIN_LOG_LEVEL=2

# Tetapkan working directory di dalam container
WORKDIR /app

# Install dependensi sistem yang dibutuhkan oleh Pillow, TensorFlow, dan tools pendukung
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Salin file requirements terlebih dahulu untuk memanfaatkan caching layer Docker
COPY requirements.txt .

# Install dependencies Python tanpa menyimpan cache pip agar image tetap ramping
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi, model, dan konfigurasi ke dalam container
COPY . .

# Port yang dibuka oleh container
EXPOSE 5000

# Health check untuk memastikan container berjalan normal
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Perintah default untuk menjalankan aplikasi menggunakan Gunicorn WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "120", "app:app"]
