"""
Definisi Endpoint RESTful API untuk Deteksi Penyakit Daun Jeruk Bali (Pomelo).

Modul ini mengelola rute:
1. GET  /               : Health check server dan informasi sistem.
2. POST /api/v1/predict : Prediksi penyakit daun menggunakan pipeline Two-Step Verification.
"""

import os
import traceback
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from services.ai_service import ai_service, InvalidImageError

# Inisialisasi Blueprint API
api_bp = Blueprint("api", __name__)

# Ekstensi file gambar yang diizinkan
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename: str) -> bool:
    """Mengecek apakah ekstensi file termasuk dalam daftar yang diperbolehkan."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@api_bp.route("/", methods=["GET"])
def health_check():
    """
    Health Check Server
    ---
    tags:
      - Sistem
    summary: Pengecekan status server dan kesehatan sistem
    description: Mengembalikan status kesehatan server dan info model yang sedang aktif.
    responses:
      200:
        description: Server berjalan normal.
        schema:
          type: object
          properties:
            status:
              type: string
              example: success
            message:
              type: string
              example: Pomelo Disease Detection API is running healthy.
            version:
              type: string
              example: 1.0.0
            architecture:
              type: string
              example: Two-Step Verification (Gatekeeper MobileNetV2 + Expert VGG16)
    """
    return jsonify({
        "status": "success",
        "message": "Pomelo Disease Detection API is running healthy.",
        "version": "1.0.0",
        "architecture": "Two-Step Verification (Gatekeeper MobileNetV2 + Expert VGG16)",
        "classes": ai_service.labels
    }), 200


@api_bp.route("/api/v1/predict", methods=["POST"])
def predict():
    """
    Klasifikasi Penyakit Daun Jeruk Bali (Two-Step Verification)
    ---
    tags:
      - Prediksi Penyakit
    summary: Mengunggah gambar daun untuk klasifikasi penyakit
    description: >
      Endpoint ini menerima gambar melalui multipart/form-data dengan key 'file'.
      Pipeline verifikasi 2 tahap akan dijalankan:
      1. Gatekeeper Model (MobileNetV2): Memverifikasi apakah gambar daun Jeruk Bali.
      2. Expert Model (VGG16): Mengklasifikasikan jenis penyakit dan rekomendasi penanganan.
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: File gambar daun jeruk bali (.jpg, .jpeg, .png, .webp)
    responses:
      200:
        description: Klasifikasi berhasil dan gambar terverifikasi sebagai daun jeruk bali.
        schema:
          type: object
          properties:
            status:
              type: string
              example: success
            message:
              type: string
              example: Klasifikasi penyakit daun jeruk bali berhasil.
            data:
              type: object
              properties:
                id_kelas:
                  type: string
                  example: Pomelo_Cephaleuros_virescens
                tingkat_keyakinan_persen:
                  type: number
                  example: 98.45
                detail_penyakit:
                  type: object
                  properties:
                    nama_ilmiah:
                      type: string
                      example: Cephaleuros virescens (Algal Spot)
                    nama_umum:
                      type: string
                      example: Bercak Ganggang
                    bahaya:
                      type: string
                      example: Sedang
                    deskripsi:
                      type: string
                      example: Bercak menonjol berwarna cokelat kehijauan pada permukaan daun...
                    penanganan:
                      type: array
                      items:
                        type: string
      400:
        description: Request tidak valid (tidak ada file, format salah, atau objek bukan daun jeruk bali).
        schema:
          type: object
          properties:
            status:
              type: string
              example: fail
            message:
              type: string
              example: Gambar tidak valid. Objek bukan daun Jeruk Bali.
      500:
        description: Kesalahan internal server saat memproses inferensi AI.
        schema:
          type: object
          properties:
            status:
              type: string
              example: error
            message:
              type: string
              example: Terjadi kesalahan internal server saat memproses gambar.
    """
    # 1. Validasi keberadaan file dalam request
    if "file" not in request.files:
        return jsonify({
            "status": "fail",
            "message": "Key 'file' tidak ditemukan dalam multipart form-data. Pastikan key input bernilai 'file'."
        }), 400

    file = request.files["file"]

    # 2. Validasi apakah ada file yang dipilih
    if file.filename == "":
        return jsonify({
            "status": "fail",
            "message": "Tidak ada file yang dipilih untuk diunggah."
        }), 400

    # 3. Validasi ekstensi file
    if not allowed_file(file.filename):
        return jsonify({
            "status": "fail",
            "message": f"Format file tidak didukung. Harap unggah file dengan format: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        }), 400

    # 4. Proses inferensi AI Two-Step Pipeline
    try:
        result = ai_service.predict(file.stream)

        return jsonify({
            "status": "success",
            "message": "Klasifikasi penyakit daun jeruk bali berhasil.",
            "data": result
        }), 200

    except InvalidImageError as e:
        # Ditangkap jika Gatekeeper / Satpam mendeteksi skor < 0.5 (Bukan daun Pomelo)
        return jsonify({
            "status": "fail",
            "message": e.message,
            "satpam_score_persen": e.gatekeeper_score
        }), 400

    except ValueError as e:
        # Ditangkap jika file bukan gambar yang valid saat dibuka oleh PIL
        return jsonify({
            "status": "fail",
            "message": f"File gambar rusak atau tidak dapat diproses: {str(e)}"
        }), 400

    except Exception as e:
        # Tangani error tak terduga
        print(f"[ERROR PREDICT] {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Terjadi kesalahan internal server saat memproses gambar: {str(e)}"
        }), 500
