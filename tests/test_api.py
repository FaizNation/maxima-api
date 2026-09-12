"""
Pengujian Otomatis API (Pytest).

File ini berisi pengujian unit/integrasi dasar untuk memvalidasi
kesehatan endpoint server pada Continuous Integration (CI) pipeline.
"""

import pytest
from app import create_app


@pytest.fixture
def client():
    """
    Fixture Pytest untuk menginisialisasi Flask test client.
    Mengatur mode TESTING ke True dan menghasilkan instance test client.
    """
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_health_check(client):
    """
    Pengujian sederhana: Memastikan endpoint GET / dapat diakses normal.
    - Status Code harus 200 (OK).
    - Status payload JSON harus bernilai 'success'.
    - Memastikan struktur data informasi kesehatan sistem tersedia.
    """
    response = client.get("/")
    
    # 1. Verifikasi HTTP Status Code
    assert response.status_code == 200

    # 2. Verifikasi Konten JSON
    data = response.get_json()
    assert data is not None
    assert data.get("status") == "success"
    assert "message" in data
    assert "version" in data
    assert "classes" in data
    assert isinstance(data.get("classes"), list)
