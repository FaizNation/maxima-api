"""
Aplikasi Utama Flask - Pomelo Disease Detection API & Maxist Chatbot.

Sistem RESTful API untuk klasifikasi penyakit daun jeruk bali (Pomelo)
menggunakan arsitektur Two-Step Verification (Gatekeeper & Expert Model)
serta Chatbot Asisten Multimodal 'Maxist' berbasis Google Gemini 1.5 Flash.

Inisialisasi:
- Flask Framework
- Cross-Origin Resource Sharing (CORS)
- Swagger UI (Flasgger)
- Registrasi Blueprint API
"""

import os
from flask import Flask, jsonify
from flask_cors import CORS
from flasgger import Swagger
from api.routes import api_bp


def create_app() -> Flask:
    """
    Factory function untuk membuat dan mengonfigurasi instance aplikasi Flask.
    """
    app = Flask(__name__)

    # Konfigurasi batas ukuran maksimal upload (16 MB)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False

    # Inisialisasi CORS untuk mengizinkan akses frontend cross-origin
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Konfigurasi Flasgger (OpenAPI 2.0 / Swagger UI)
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": "apispec",
                "route": "/apispec.json",
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/apidocs/"
    }

    swagger_template = {
        "swagger": "2.0",
        "info": {
            "title": "Pomelo Leaf Disease Detection & Maxist AI Assistant API",
            "description": (
                "RESTful AI Microservice cerdas untuk platform MAXIMA:\n\n"
                "1. **Two-Step Verification Disease Detection:**\n"
                "   - **Gatekeeper (MobileNetV2 Sigmoid):** Verifikasi keaslian daun jeruk bali (skor >= 0.5).\n"
                "   - **Expert Model (VGG16 Softmax):** Diagnosis 4 kondisi daun (*Algal Spot*, *Healthy*, *Leaf Miner*, *Orange Mold*) diperkaya rekomendasi solusi.\n\n"
                "2. **Maxist Multimodal AI Assistant (Google Gemini 1.5 Flash):**\n"
                "   - Konsultasi interaktif berbahasa Indonesia ramah petani.\n"
                "   - Analisis citra daun langsung via `image_url` secara multimodal.\n"
                "   - Integrasi konteks riwayat scan dari database Express.js (`db_context`)."
            ),
            "version": "1.1.0",
            "contact": {
                "name": "MAXIMA AI Engineering Team",
                "email": "developer@pomelo-ai.local"
            },
            "license": {
                "name": "MIT License",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        "tags": [
            {
                "name": "Sistem",
                "description": "Endpoint manajemen status dan kesehatan server"
            },
            {
                "name": "Prediksi Penyakit",
                "description": "Endpoint inferensi AI Two-Step Verification"
            },
            {
                "name": "Chatbot Asisten Maxist",
                "description": "Endpoint percakapan AI multimodal Maxist (Google Gemini 1.5 Flash)"
            }
        ]
    }

    Swagger(app, config=swagger_config, template=swagger_template)

    # Registrasi blueprint routing
    app.register_blueprint(api_bp)

    # Global Error Handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({
            "status": "fail",
            "message": "Endpoint tidak ditemukan. Silakan cek dokumentasi di /apidocs."
        }), 404

    @app.errorhandler(405)
    def method_not_allowed_error(error):
        return jsonify({
            "status": "fail",
            "message": "Metode HTTP tidak diizinkan untuk endpoint ini."
        }), 405

    @app.errorhandler(413)
    def request_entity_too_large_error(error):
        return jsonify({
            "status": "fail",
            "message": "Ukuran file terlalu besar. Batas maksimal ukuran gambar adalah 16 MB."
        }), 413

    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({
            "status": "error",
            "message": "Terjadi kesalahan internal pada server aplikasi."
        }), 500

    return app


# Instance aplikasi untuk Gunicorn / Production WSGI Server
app = create_app()

if __name__ == "__main__":
    # Port dari environment variable (default: 5000)
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")

    print("=" * 70)
    print("  POMELO LEAF DISEASE DETECTION & MAXIST AI API")
    print(f"  Server berjalan di : http://{host}:{port}")
    print(f"  Dokumentasi Swagger: http://localhost:{port}/apidocs")
    print("=" * 70)

    app.run(host=host, port=port, debug=False)
