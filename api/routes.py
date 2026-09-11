"""
Definisi Endpoint RESTful API untuk Deteksi Penyakit Daun Jeruk Bali (Pomelo)
serta Asisten AI Multimodal 'Maxist'.

Modul ini mengelola rute:
1. GET  /               : Health check server dan informasi sistem.
2. POST /api/v1/predict : Prediksi penyakit daun menggunakan pipeline Two-Step Verification.
3. POST /api/v1/chat    : Chatbot cerdas Maxist (Gemini 1.5 Flash multimodal).
"""

import os
import traceback
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from services.ai_service import ai_service, InvalidImageError
from services.chat_service import (
    generate_maxist_response,
    ChatServiceError,
    MissingApiKeyError,
    ImageDownloadError
)

# Inisialisasi Blueprint API
api_bp = Blueprint("api", __name__)

# Ekstensi file gambar yang diizinkan untuk endpoint prediksi
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
              example: 1.1.0
            architecture:
              type: string
              example: Two-Step Verification (Gatekeeper MobileNetV2 + Expert VGG16) & Maxist Chatbot (Gemini 1.5 Flash)
    """
    return jsonify({
        "status": "success",
        "message": "Pomelo Disease Detection API is running healthy.",
        "version": "1.1.0",
        "architecture": "Two-Step Verification (Gatekeeper MobileNetV2 + Expert VGG16) & Maxist Chatbot (Gemini 1.5 Flash)",
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


@api_bp.route("/api/v1/chat", methods=["POST"])
def chat():
    """
    Chatbot Cerdas Maxist (Multimodal Gemini 1.5 Flash)
    ---
    tags:
      - Chatbot Asisten Maxist
    summary: Konsultasi percakapan multimodal dengan asisten AI Maxist
    description: >
      Menerima pesan pertanyaan petani, riwayat percakapan sebelumnya, konteks database dari Express.js,
      serta URL gambar foto scan daun (opsional) untuk dianalisis secara multimodal oleh Gemini 1.5 Flash.
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        description: Payload percakapan dengan asisten Maxist
        schema:
          type: object
          required:
            - message
          properties:
            message:
              type: string
              description: Pesan atau pertanyaan dari petani/pengguna.
              example: "Bagaimana cara mengatasi bercak ganggang pada daun jeruk bali saya?"
            history:
              type: array
              description: Riwayat percakapan sebelumnya.
              items:
                type: object
                properties:
                  role:
                    type: string
                    example: user
                  parts:
                    type: array
                    items:
                      type: string
                    example: ["Halo Maxist"]
              example:
                - role: user
                  parts: ["Halo Maxist, saya petani jeruk bali."]
                - role: model
                  parts: ["Halo Bapak/Ibu Petani! Ada yang bisa Maxist bantu seputar kebun jeruk bali Anda hari ini?"]
            db_context:
              type: string
              description: Konteks riwayat scan atau data tanaman dari Express.js.
              example: "Hasil scan terakhir: Terindikasi Bercak Ganggang (Cephaleuros virescens) dengan tingkat keyakinan 98.45%."
            image_url:
              type: string
              description: URL gambar daun jeruk bali yang ingin dibahas bersama asisten.
              example: "https://images.unsplash.com/photo-1542273917363-3b1817f69a2d"
    responses:
      200:
        description: Balasan berhasil didapatkan dari asisten Maxist.
        schema:
          type: object
          properties:
            status:
              type: string
              example: success
            data:
              type: object
              properties:
                reply:
                  type: string
                  example: "Halo Bapak/Ibu Petani! Untuk mengatasi bercak ganggang pada daun jeruk bali..."
      400:
        description: Request tidak valid (parameter message tidak ada, format JSON salah, atau gagal download gambar).
        schema:
          type: object
          properties:
            status:
              type: string
              example: fail
            message:
              type: string
              example: Parameter 'message' wajib diisi dan tidak boleh kosong.
      500:
        description: Kesalahan internal server atau kegagalan API Gemini.
        schema:
          type: object
          properties:
            status:
              type: string
              example: error
            message:
              type: string
              example: Terjadi kesalahan saat memproses percakapan dengan AI.
    """
    # 1. Validasi format Content-Type JSON
    if not request.is_json:
        return jsonify({
            "status": "fail",
            "message": "Content-Type request harus berupa 'application/json'."
        }), 400

    # 2. Parsing payload JSON
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "status": "fail",
            "message": "Format payload JSON tidak valid atau kosong."
        }), 400

    # 3. Validasi parameter 'message' yang wajib diisi
    message = data.get("message")
    if not message or not str(message).strip():
        return jsonify({
            "status": "fail",
            "message": "Parameter 'message' wajib diisi dan tidak boleh kosong."
        }), 400

    # 4. Ambil parameter opsional (history, db_context, image_url)
    history = data.get("history", [])
    db_context = data.get("db_context")
    image_url = data.get("image_url")

    # 5. Eksekusi pemanggilan ke layanan AI Chatbot Maxist
    try:
        reply = generate_maxist_response(
            message=str(message).strip(),
            history=history,
            db_context=db_context,
            image_url=image_url
        )

        return jsonify({
            "status": "success",
            "data": {
                "reply": reply
            }
        }), 200

    except MissingApiKeyError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    except ImageDownloadError as e:
        return jsonify({
            "status": "fail",
            "message": str(e)
        }), 400

    except ValueError as e:
        return jsonify({
            "status": "fail",
            "message": str(e)
        }), 400

    except ChatServiceError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    except Exception as e:
        print(f"[ERROR CHAT] {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Terjadi kesalahan internal server saat memproses pesan: {str(e)}"
        }), 500
