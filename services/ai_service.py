"""
Modul AI Service untuk Klasifikasi Penyakit Daun Jeruk Bali (Pomelo).

Mengimplementasikan arsitektur 'Two-Step Verification':
1. Gatekeeper Model (MobileNetV2): Menyaring apakah gambar adalah Daun Pomelo atau Bukan.
2. Expert Model (VGG16): Mengklasifikasikan jenis penyakit dari 4 kelas daun jeruk bali.
"""

import os
import json
import numpy as np
from PIL import Image
from utils.knowledge_base import get_disease_info

# Inisialisasi Keras / TensorFlow loader
try:
    import keras
    _load_model = keras.models.load_model
except ImportError:
    import tensorflow as tf
    _load_model = tf.keras.models.load_model


class InvalidImageError(Exception):
    """
    Custom Exception yang dilemparkan ketika gambar diverifikasi oleh model Satpam
    bukan merupakan daun jeruk bali (skor sigmoid < 0.5).
    """
    def __init__(self, message: str = "Gambar tidak valid. Objek bukan daun Jeruk Bali.", gatekeeper_score: float = 0.0):
        super().__init__(message)
        self.message = message
        self.gatekeeper_score = gatekeeper_score


class AIService:
    """
    Kelas Singleton untuk memuat model AI dan melayani inferensi Two-Step Verification.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AIService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Tentukan path file model dan label
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.satpam_model_path = os.getenv("SATPAM_MODEL_PATH", os.path.join(base_dir, "model_satpam_pomelo.h5"))
        self.disease_model_path = os.getenv("DISEASE_MODEL_PATH", os.path.join(base_dir, "pomelo_disease_model.h5"))
        self.labels_path = os.getenv("LABELS_PATH", os.path.join(base_dir, "pomelo_labels.json"))

        # Validasi keberadaan file penting
        self._validate_files()

        # Load label JSON di awal saat modul di-load
        print("[AI Service] Memuat label penyakit...")
        with open(self.labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)
        print(f"[AI Service] Label berhasil dimuat: {self.labels}")

        # Load kedua model .h5 di awal saat inisialisasi agar inferensi responsif
        print(f"[AI Service] Memuat model Satpam (Gatekeeper) dari {self.satpam_model_path}...")
        self.satpam_model = _load_model(self.satpam_model_path, compile=False)
        print("[AI Service] Model Satpam berhasil dimuat.")

        print(f"[AI Service] Memuat model Pakar Penyakit (Expert) dari {self.disease_model_path}...")
        self.disease_model = _load_model(self.disease_model_path, compile=False)
        print("[AI Service] Model Pakar Penyakit berhasil dimuat.")

        self._initialized = True

    def _validate_files(self):
        """Memastikan semua file model dan konfigurasi tersedia."""
        for path, name in [
            (self.satpam_model_path, "Model Satpam"),
            (self.disease_model_path, "Model Pakar Penyakit"),
            (self.labels_path, "Label Penyakit JSON")
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File {name} tidak ditemukan pada lokasi: {path}")

    def preprocess_image(self, file_stream) -> np.ndarray:
        """
        Fungsi preprocessing gambar:
        - Buka gambar menggunakan Pillow.
        - Pastikan format gambar adalah RGB (mengonversi RGBA/Grayscale).
        - Resize gambar ke resolusi 224x224 piksel.
        - Ubah ke format numpy array dengan tipe float32.
        - Tambahkan dimensi batch (expand_dims) sehingga bentuknya menjadi (1, 224, 224, 3).
        - Tanpa background remover sesuai spesifikasi.
        """
        try:
            if hasattr(file_stream, "seek"):
                file_stream.seek(0)

            image = Image.open(file_stream)

            # Konversi jika format bukan RGB (misal RGBA atau Grayscale)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize ke 224x224
            image = image.resize((224, 224), Image.Resampling.BILINEAR)

            # Konversi ke numpy array
            img_array = np.array(image, dtype=np.float32)

            # Tambahkan dimensi batch (1, 224, 224, 3)
            img_array = np.expand_dims(img_array, axis=0)

            return img_array
        except Exception as e:
            raise ValueError(f"Gagal memproses gambar: {str(e)}")

    def predict(self, file_stream) -> dict:
        """
        Menjalankan pipeline Two-Step Verification:
        1. Prediksi menggunakan model_satpam_pomelo.h5.
           - Jika output < 0.5: Lemparkan InvalidImageError ("Gambar tidak valid. Objek bukan daun Jeruk Bali.").
        2. Jika output >= 0.5:
           - Teruskan array gambar ke model pakar (pomelo_disease_model.h5).
           - Ambil indeks probabilitas tertinggi (argmax).
           - Petakan ke label dari JSON dan data penanganan dari knowledge base.
           - Kembalikan dictionary data prediksi lengkap.
        """
        # 1. Preprocessing gambar
        img_array = self.preprocess_image(file_stream)

        # 2. Step 1: Gatekeeper / Model Satpam (MobileNetV2 Sigmoid)
        satpam_pred = self.satpam_model.predict(img_array, verbose=0)
        satpam_score = float(satpam_pred[0][0])

        # Ambang batas verifikasi satpam (0.5)
        if satpam_score < 0.5:
            raise InvalidImageError(
                message="Gambar tidak valid. Objek bukan daun Jeruk Bali.",
                gatekeeper_score=round(satpam_score * 100, 2)
            )

        # 3. Step 2: Expert Model (VGG16 Softmax)
        disease_pred = self.disease_model.predict(img_array, verbose=0)
        class_idx = int(np.argmax(disease_pred[0]))
        confidence_percent = round(float(disease_pred[0][class_idx]) * 100, 2)

        # Cocokkan index argmax dengan label JSON
        if 0 <= class_idx < len(self.labels):
            label_id = self.labels[class_idx]
        else:
            label_id = "Unknown"

        # Ambil data detail penyakit dan penanganannya dari knowledge base
        detail_penyakit = get_disease_info(label_id)

        # Probabilitas untuk semua kelas penyakit
        all_probabilities = {}
        for idx, lbl in enumerate(self.labels):
            if idx < len(disease_pred[0]):
                all_probabilities[lbl] = round(float(disease_pred[0][idx]) * 100, 2)

        return {
            "id_kelas": label_id,
            "tingkat_keyakinan_persen": confidence_percent,
            "detail_penyakit": detail_penyakit,
            "verifikasi_satpam": {
                "lulus": True,
                "skor_keyakinan_daun_persen": round(satpam_score * 100, 2)
            },
            "probabilitas_semua_kelas": all_probabilities
        }


# Inisialisasi instance AI service di level modul agar model siap saat aplikasi di-import
ai_service = AIService()
