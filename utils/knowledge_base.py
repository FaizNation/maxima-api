"""
Modul Knowledge Base (Buku Pintar) Penyakit Daun Jeruk Bali (Pomelo).

Modul ini memuat basis pengetahuan mengenai berbagai kondisi dan penyakit
daun jeruk bali, mencakup nama ilmiah, nama umum, tingkat bahaya, deskripsi,
serta rekomendasi penanganan komprehensif untuk petani maupun pengguna umum.
"""

from typing import Dict, Any, List

# Basis pengetahuan terstruktur untuk setiap kelas klasifikasi daun jeruk bali
DISEASE_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "Pomelo_Cephaleuros_virescens": {
        "nama_ilmiah": "Cephaleuros virescens (Algal Spot)",
        "nama_umum": "Bercak Ganggang",
        "bahaya": "Sedang",
        "deskripsi": (
            "Penyakit bercak ganggang disebabkan oleh alga parasit Cephaleuros virescens. "
            "Gejala ditandai dengan munculnya bercak melingkar yang agak menonjol seperti beludru "
            "berwarna hijau kelabu hingga jingga/karat pada permukaan atas daun jeruk bali. "
            "Dapat mengganggu fotosintesis dan menyebabkan daun menguning serta rontok prematur."
        ),
        "penanganan": [
            "Pangkas daun dan ranting yang terinfeksi berat lalu bakar atau musnahkan jauh dari kebun.",
            "Semprotkan fungisida berbahan aktif tembaga (copper-based fungicide seperti tembaga oksiklorida) secara merata.",
            "Lakukan pemangkasan cabang secara berkala untuk meningkatkan sirkulasi udara dan penetrasi sinar matahari ke dalam tajuk tanaman.",
            "Tingkatkan vigor tanaman melalui pemupukan seimbang, terutama kalium dan unsur mikro."
        ]
    },
    "Pomelo_Leaf_Miner": {
        "nama_ilmiah": "Phyllocnistis citrella (Citrus Leaf Miner)",
        "nama_umum": "Hama Pengorok Daun",
        "bahaya": "Tinggi",
        "deskripsi": (
            "Disebabkan oleh larva ngengat kecil Phyllocnistis citrella yang memakan lapisan epidermis "
            "dan mesofil daun muda jeruk bali. Gejala khas berupa lorong-lorong berliku-liku (galeri) "
            "berwarna keperakan transparan. Daun terinfeksi menjadi keriting, menggulung, kerdil, "
            "dan sangat rentan terhadap infeksi sekunder seperti kanker jeruk (Citrus Canker)."
        ),
        "penanganan": [
            "Petik dan bakar daun yang menunjukkan adanya lorong aktif atau larva pengorok.",
            "Semprotkan insektisida nabati berbahan minyak mimba (neem oil) atau ekstrak tembakau pada tunas-tunas muda.",
            "Pasang perangkap lekat kuning (yellow sticky trap) di area perkebunan untuk menangkap ngengat dewasa.",
            "Gunakan insektisida sistemik berbahan aktif abamektin atau imidakloprid sesuai anjuran jika intensitas serangan sudah meluas."
        ]
    },
    "Pomelo_Orange_Mold": {
        "nama_ilmiah": "Orange Mold (Infeksi Kapang Oranye)",
        "nama_umum": "Jamur Kapang Oranye",
        "bahaya": "Tinggi",
        "deskripsi": (
            "Infeksi jamur patogenik yang memicu pertumbuhan massa spora tebal berwarna oranye kemerahan "
            "pada permukaan daun maupun tangkai daun jeruk bali. Penyakit ini berkembang sangat cepat pada "
            "kondisi lingkungan dengan kelembapan tinggi dan drainase buruk, menyebabkan nekrosis jaringan, "
            "klorosis hebat, dan kerontokan daun massal."
        ),
        "penanganan": [
            "Segera isolasi tanaman atau ranting yang terinfeksi untuk mencegah penyebaran spora ke pohon lain.",
            "Pangkas dan buang bagian daun yang menunjukkan tanda-tanda pembentukan spora kapang oranye.",
            "Aplikasi fungisida sistemik spektrum luas (seperti kelompok triazol atau strobilurin) sesuai dosis.",
            "Perbaiki sanitasi kebun, bersihkan gulma di bawah kanopi, dan kurangi frekuensi penyiraman berlebih.",
            "Hindari menyiram daun langsung dari atas (overhead watering) untuk menekan kelembapan mikro."
        ]
    },
    "Pomelo_Healthy": {
        "nama_ilmiah": "Citrus maxima (Healthy Foliage)",
        "nama_umum": "Daun Sehat",
        "bahaya": "Aman",
        "deskripsi": (
            "Daun jeruk bali berada dalam kondisi prima, segar, berwarna hijau merata, dan memiliki turgor yang baik. "
            "Tidak terdeteksi adanya infeksi ganggang, jamur kapang, maupun serangan hama perusak jaringan daun."
        ),
        "penanganan": [
            "Lanjutkan program pemupukan berimbang (kombinasi pupuk organik dan NPK) sesuai fase vegetatif tanaman.",
            "Pertahankan jadwal irigasi secara teratur dengan memastikan drainase tanah tetap lancar tanpa genangan air.",
            "Jaga sanitasi kebun dan lakukan inspeksi visual rutin setiap minggu untuk deteksi dini hama dan penyakit.",
            "Berikan perlakuan mulsa di sekitar piringan pohon untuk menjaga kelembapan tanah yang stabil."
        ]
    }
}


def get_disease_info(disease_key: str) -> Dict[str, Any]:
    """
    Mengambil data detail penyakit dari basis pengetahuan berdasarkan ID kelas.
    
    Args:
        disease_key (str): Kunci/ID kelas penyakit (misal: 'Pomelo_Healthy')
        
    Returns:
        Dict[str, Any]: Dictionary berisi informasi detail penyakit.
    """
    if disease_key in DISEASE_KNOWLEDGE_BASE:
        return DISEASE_KNOWLEDGE_BASE[disease_key]
    
    # Fallback jika ada kelas di luar ekspektasi
    return {
        "nama_ilmiah": disease_key,
        "nama_umum": "Kondisi Tidak Dikenal",
        "bahaya": "Tidak Diketahui",
        "deskripsi": "Informasi untuk kelas ini belum terdaftar di basis pengetahuan.",
        "penanganan": ["Konsultasikan dengan ahli patologi tanaman terdekat."]
    }
