"""
Modul Layanan AI Chatbot 'Maxist' (Multimodal AI Assistant).

Menggunakan Google Gemini API (model: gemini-1.5-flash) untuk melayani
tanya jawab cerdas seputar budidaya, perawatan, dan penanganan penyakit
daun jeruk bali (Pomelo) bagi para petani platform MAXIMA.
"""

import os
import io
import requests
from typing import List, Dict, Any, Optional
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# Muat environment variables dari file .env
load_dotenv()

# System Instruction khusus untuk membentuk persona dan batasan jawaban Maxist
SYSTEM_INSTRUCTION = (
    "Kamu adalah Maxist, asisten AI cerdas untuk petani jeruk bali (Pomelo) dari platform MAXIMA. "
    "Jawab dengan bahasa Indonesia yang ramah, sopan, dan mudah dipahami petani. "
    "Jika diberikan [Konteks Database], gunakan info tersebut dengan natural tanpa menyebutkan kata 'sistem' atau 'database'."
)


class ChatServiceError(Exception):
    """Base exception untuk kesalahan pada Chat Service."""
    pass


class MissingApiKeyError(ChatServiceError):
    """Dilemparkan ketika GEMINI_API_KEY tidak ditemukan di environment variables."""
    pass


class ImageDownloadError(ChatServiceError):
    """Dilemparkan ketika pengunduhan gambar dari image_url mengalami kegagalan."""
    pass


def get_gemini_model():
    """
    Mengambil dan mengonfigurasi instance model GenerativeModel Gemini 1.5 Flash.
    
    Returns:
        genai.GenerativeModel: Model Gemini yang telah dikonfigurasi dengan system instruction.
        
    Raises:
        MissingApiKeyError: Jika GEMINI_API_KEY tidak tersedia di environment variables.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        raise MissingApiKeyError(
            "GEMINI_API_KEY belum dikonfigurasi di environment variables. "
            "Harap tambahkan GEMINI_API_KEY pada file .env atau variabel lingkungan sistem."
        )

    genai.configure(api_key=api_key.strip())
    return genai.GenerativeModel(
        model_name="gemini-3.6-flash",
        system_instruction=SYSTEM_INSTRUCTION
    )


def _download_image(image_url: str) -> Image.Image:
    """
    Mengunduh gambar dari URL ke dalam memori buffer dan membukanya dengan PIL Image.
    
    Args:
        image_url (str): Tautan URL gambar.
        
    Returns:
        PIL.Image.Image: Objek gambar dalam format RGB.
        
    Raises:
        ImageDownloadError: Jika URL tidak valid, timeout, atau data bukan citra.
    """
    if not image_url.startswith(("http://", "https://")):
        raise ImageDownloadError(f"URL gambar tidak valid: '{image_url}'. URL harus diawali http:// atau https://.")

    headers = {
        "User-Agent": "Maxima-AI-Microservice/1.0 (Pomelo Disease Assistant)"
    }

    try:
        response = requests.get(image_url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise ImageDownloadError(f"Waktu koneksi habis (timeout) saat mengunduh gambar dari: {image_url}")
    except requests.exceptions.RequestException as e:
        raise ImageDownloadError(f"Gagal mengunduh gambar dari URL: {str(e)}")

    try:
        image = Image.open(io.BytesIO(response.content))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise ImageDownloadError(f"Konten dari URL bukan format gambar yang valid: {str(e)}")


def _normalize_history(raw_history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Menormalisasi format riwayat percakapan agar sesuai dengan skema Google Gemini:
    [{'role': 'user'|'model', 'parts': ['isi pesan']}]
    """
    if not raw_history or not isinstance(raw_history, list):
        return []

    formatted = []
    for item in raw_history:
        if not isinstance(item, dict):
            continue

        role = item.get("role", "user")
        # Ubah role 'assistant' menjadi 'model' sesuai konvensi Gemini
        if role == "assistant":
            role = "model"
        elif role not in ("user", "model"):
            role = "user"

        parts: List[str] = []
        if "parts" in item and isinstance(item["parts"], list):
            parts = [str(p) for p in item["parts"]]
        elif "content" in item and isinstance(item["content"], str):
            parts = [item["content"]]
        elif "text" in item and isinstance(item["text"], str):
            parts = [item["text"]]
        elif "message" in item and isinstance(item["message"], str):
            parts = [item["message"]]

        if parts:
            formatted.append({"role": role, "parts": parts})

    return formatted


def generate_maxist_response(
    message: str,
    history: Optional[List[Dict[str, Any]]] = None,
    db_context: Optional[str] = None,
    image_url: Optional[str] = None
) -> str:
    """
    Menghasilkan respon percakapan asisten cerdas Maxist menggunakan Gemini 1.5 Flash.

    Args:
        message (str): Pesan teks pertanyaan dari petani / user (wajib).
        history (list, opsional): Riwayat percakapan sebelumnya.
        db_context (str, opsional): Konteks riwayat database dari backend Express.
        image_url (str, opsional): URL foto daun jika user ingin mendiskusikan hasil scan.

    Returns:
        str: Respon teks balasan dari asisten Maxist.
        
    Raises:
        MissingApiKeyError: Jika API key belum disetel.
        ImageDownloadError: Jika pengunduhan gambar gagal.
        ChatServiceError: Jika terjadi error saat memanggil API Gemini.
    """
    if not message or not str(message).strip():
        raise ValueError("Parameter 'message' tidak boleh kosong.")

    # 1. Inisialisasi model Gemini 1.5 Flash
    model = get_gemini_model()

    # 2. Susun prompt teks yang mengintegrasikan [Konteks Database] jika ada
    text_content = ""
    if db_context and str(db_context).strip():
        text_content = (
            f"[Konteks Database]\n{str(db_context).strip()}\n\n"
            f"Pertanyaan Petani:\n{message.strip()}"
        )
    else:
        text_content = message.strip()

    # 3. Jalankan logika Multimodal (jika image_url disertakan)
    if image_url and str(image_url).strip():
        # Unduh gambar ke dalam memori
        pil_image = _download_image(str(image_url).strip())

        try:
            # Kirim gambar dan teks langsung ke model Gemini 1.5 Flash
            response = model.generate_content([pil_image, text_content])
            if not response.text:
                return "Maaf, Maxist belum dapat memproses gambar tersebut saat ini. Silakan coba beberapa saat lagi."
            return response.text.strip()
        except Exception as e:
            raise ChatServiceError(f"Terjadi kegagalan saat analisis multimodal Gemini: {str(e)}")

    # 4. Jalankan percakapan teks biasa dengan dukungan history
    formatted_history = _normalize_history(history)

    try:
        chat = model.start_chat(history=formatted_history)
        response = chat.send_message(text_content)
        if not response.text:
            return "Maaf, Maxist tidak dapat merespon saat ini. Silakan ulangi pertanyaan Anda."
        return response.text.strip()
    except Exception as e:
        raise ChatServiceError(f"Terjadi kesalahan saat berkomunikasi dengan model Gemini: {str(e)}")
