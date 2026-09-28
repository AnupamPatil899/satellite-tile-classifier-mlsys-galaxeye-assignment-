import os
import io
import time
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.config import settings
from app.database import init_db

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_database():
    init_db()

def create_test_image_bytes(color=(100, 200, 100), size=(64, 64), fmt="PNG"):
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["offline_mode"] is True
    assert "runtime_engine" in data
    assert data["model_loaded"] is True
    assert data["database_connected"] is True

def test_dashboard_endpoint():
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "GalaxEye" in response.text
    assert "Satellite Intelligence Console" in response.text

def test_classify_endpoint_success():
    # Use distinct color so hash is unique
    unique_color = (int(time.time() * 1000) % 250, 145, 67)
    img_bytes = create_test_image_bytes(color=unique_color)
    files = {"file": ("unique_forest_tile.png", img_bytes, "image/png")}
    
    response = client.post("/api/v1/tiles/classify", files=files)
    assert response.status_code == 201
    data = response.json()
    
    assert "id" in data
    assert "sha256_hash" in data
    assert data["original_filename"] == "unique_forest_tile.png"
    assert data["predicted_class"] in settings.CLASSES
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["margin"] <= 1.0
    assert 0.0 <= data["entropy"] <= 1.0
    assert data["status"] in ["CONFIDENT", "UNCERTAIN"]
    assert len(data["class_probabilities"]) == len(settings.CLASSES)
    assert data["inference_latency_ms"] > 0

def test_deduplication_via_sha256():
    unique_color = (88, 12, (int(time.time() * 1000) + 50) % 250)
    img_bytes = create_test_image_bytes(color=unique_color)
    files = {"file": ("sea_tile.png", img_bytes, "image/png")}
    
    res1 = client.post("/api/v1/tiles/classify", files=files)
    assert res1.status_code == 201
    tile1 = res1.json()
    assert tile1.get("is_duplicate", False) is False

    files2 = {"file": ("copy_sea_tile.png", img_bytes, "image/png")}
    res2 = client.post("/api/v1/tiles/classify", files=files2)
    assert res2.status_code == 201
    tile2 = res2.json()
    
    assert tile2["sha256_hash"] == tile1["sha256_hash"]
    assert tile2["id"] == tile1["id"]
    assert tile2["is_duplicate"] is True

def test_batch_classification():
    t = int(time.time() * 1000) % 200
    files = [
        ("files", ("batch_1.png", create_test_image_bytes(color=(t, 20, 30)), "image/png")),
        ("files", ("batch_2.png", create_test_image_bytes(color=(t+1, 50, 60)), "image/png")),
        ("files", ("batch_3.png", create_test_image_bytes(color=(t+2, 80, 90)), "image/png")),
    ]
    response = client.post("/api/v1/tiles/batch", files=files)
    assert response.status_code == 201
    data = response.json()
    assert data["batch_size"] == 3
    assert data["inserted_count"] >= 1
    assert "runtime_engine" in data
    assert len(data["results"]) == 3

def test_active_learning_human_correction():
    t = int(time.time() * 1000) % 200
    img_bytes = create_test_image_bytes(color=(150, t, 30))
    res = client.post(
        "/api/v1/tiles/classify",
        files={"file": ("border_tile.png", img_bytes, "image/png")}
    )
    tile_id = res.json()["id"]

    correct_payload = {
        "verified_label": "River",
        "analyst_notes": "Ground inspection verified seasonal riverbed"
    }
    res_correct = client.post(f"/api/v1/tiles/{tile_id}/correct", json=correct_payload)
    assert res_correct.status_code == 200
    updated_tile = res_correct.json()
    assert updated_tile["status"] == "VERIFIED"
    assert updated_tile["verified_label"] == "River"
    assert updated_tile["analyst_notes"] == correct_payload["analyst_notes"]

    res_query = client.get("/api/v1/tiles?status=VERIFIED")
    assert res_query.status_code == 200
    ids = [t["id"] for t in res_query.json()["results"]]
    assert tile_id in ids

def test_tile_image_streaming():
    t = int(time.time() * 1000) % 200
    img_bytes = create_test_image_bytes(color=(200, 100, t))
    res = client.post(
        "/api/v1/tiles/classify",
        files={"file": ("stream_tile.png", img_bytes, "image/png")}
    )
    tile_id = res.json()["id"]

    img_res = client.get(f"/api/v1/tiles/{tile_id}/image")
    assert img_res.status_code == 200
    assert len(img_res.content) == len(img_bytes)

def test_query_tiles_with_filters():
    res = client.get("/api/v1/tiles?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert "total" in data
    assert isinstance(data["results"], list)

def test_classify_invalid_file_extension():
    files = {"file": ("malicious.exe", b"binary content", "application/octet-stream")}
    response = client.post("/api/v1/tiles/classify", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

def test_classify_corrupted_image():
    files = {"file": ("corrupt.png", b"corrupted bytes", "image/png")}
    response = client.post("/api/v1/tiles/classify", files=files)
    assert response.status_code == 400
    assert "Corrupt or invalid image data" in response.json()["detail"]

def test_system_stats():
    res = client.get("/api/v1/stats")
    assert res.status_code == 200
    stats = res.json()
    assert "total_tiles" in stats
    assert "human_corrections_recorded" in stats
    assert "avg_entropy" in stats
