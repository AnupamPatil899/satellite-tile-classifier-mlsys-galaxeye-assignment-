import os
import io
import uuid
import time
import hashlib
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, status
from fastapi.responses import HTMLResponse, FileResponse
from PIL import Image

from app.config import settings
from app.database import (
    init_db, insert_tile_record, bulk_insert_tiles,
    query_tiles, get_tile_by_id, get_system_stats,
    update_tile_verification
)
from app.schemas import (
    ClassificationResult, QueryResponse, HealthResponse,
    SystemStatsResponse, CorrectionRequest, BatchClassificationResponse
)
from app.classifier import classifier

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="High-Throughput Offline Satellite Tile Classifier with Multi-Dimensional Uncertainty Quantification & Active Learning",
    lifespan=lifespan
)

@app.get("/", include_in_schema=False)
def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "runtime_engine": classifier.runtime_engine,
        "status": "online (offline edge deployment)",
        "dashboard_url": "/dashboard",
        "docs_url": "/docs"
    }

@app.get("/dashboard", response_class=HTMLResponse, tags=["Analyst Interface"])
def get_dashboard():
    """
    Renders the embedded, zero-dependency offline analyst dashboard.
    """
    html_path = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h3>Dashboard template not found.</h3>", status_code=404)
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get(f"{settings.API_PREFIX}/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    stats = get_system_stats()
    return HealthResponse(
        status="HEALTHY",
        offline_mode=True,
        runtime_engine=classifier.runtime_engine,
        model_version=classifier.model_version,
        model_loaded=classifier.is_ready(),
        database_connected=True,
        total_tiles_processed=stats["total_tiles"],
        timestamp=datetime.utcnow().isoformat()
    )

@app.get(f"{settings.API_PREFIX}/stats", response_model=SystemStatsResponse, tags=["Monitoring"])
def system_statistics():
    stats = get_system_stats()
    return SystemStatsResponse(**stats)

@app.post(
    f"{settings.API_PREFIX}/tiles/classify",
    response_model=ClassificationResult,
    status_code=status.HTTP_201_CREATED,
    tags=["Core Ingestion & Inference"]
)
async def classify_tile(
    file: UploadFile = File(..., description="Satellite image tile (.png, .jpg, .tif)")
):
    """
    CORE ENDPOINT (Part 2 Working Slice):
    1. Validates image integrity and format.
    2. Computes SHA-256 for deterministic deduplication.
    3. Persists tile to disk.
    4. Executes CPU inference (ONNX Runtime / PyTorch).
    5. Computes Softmax Confidence, Prediction Margin, and Shannon Entropy.
    6. Tags status ('CONFIDENT' vs 'UNCERTAIN').
    7. Stores result in SQLite WAL catalog.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed: {list(settings.ALLOWED_EXTENSIONS)}"
        )
        
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )
        
    sha256 = hashlib.sha256(content).hexdigest()
    
    try:
        image = Image.open(io.BytesIO(content))
        image.verify()
        image = Image.open(io.BytesIO(content))
        width, height = image.size
        channels = len(image.getbands())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupt or invalid image data: {str(e)}"
        )
        
    tile_id = str(uuid.uuid4())
    stored_filename = f"{sha256}{ext}"
    stored_path = os.path.join(settings.STORAGE_DIR, stored_filename)
    if not os.path.exists(stored_path):
        with open(stored_path, "wb") as f:
            f.write(content)
            
    prediction = classifier.predict(image)
    
    record = {
        "id": tile_id,
        "sha256_hash": sha256,
        "original_filename": file.filename,
        "stored_path": stored_path,
        "file_size_bytes": len(content),
        "image_width": width,
        "image_height": height,
        "image_channels": channels,
        "predicted_class": prediction["predicted_class"],
        "confidence": prediction["confidence"],
        "margin": prediction["margin"],
        "entropy": prediction["entropy"],
        "class_probabilities": prediction["class_probabilities"],
        "status": prediction["status"],
        "model_version": prediction["model_version"],
        "inference_latency_ms": prediction["inference_latency_ms"],
        "created_at": datetime.utcnow().isoformat()
    }
    
    saved_record = insert_tile_record(record)
    return saved_record

@app.post(
    f"{settings.API_PREFIX}/tiles/batch",
    response_model=BatchClassificationResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["High-Throughput Burst Ingestion"]
)
async def batch_classify_tiles(
    files: List[UploadFile] = File(..., description="List of satellite tiles for vectorized batch inference")
):
    """
    HIGH-THROUGHPUT BATCH INGESTION:
    Processes burst satellite downlinks in a single vectorized CPU forward pass.
    Bulk-inserts records into SQLite using a single WAL transaction.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    start_all = time.perf_counter()
    images = []
    file_metadata = []

    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            continue
        content = await f.read()
        if len(content) == 0:
            continue
        sha256 = hashlib.sha256(content).hexdigest()
        try:
            img = Image.open(io.BytesIO(content))
            width, height = img.size
            channels = len(img.getbands())
            
            stored_filename = f"{sha256}{ext}"
            stored_path = os.path.join(settings.STORAGE_DIR, stored_filename)
            if not os.path.exists(stored_path):
                with open(stored_path, "wb") as out_f:
                    out_f.write(content)

            images.append(img)
            file_metadata.append({
                "id": str(uuid.uuid4()),
                "sha256_hash": sha256,
                "original_filename": f.filename,
                "stored_path": stored_path,
                "file_size_bytes": len(content),
                "image_width": width,
                "image_height": height,
                "image_channels": channels,
                "created_at": datetime.utcnow().isoformat()
            })
        except Exception:
            continue

    if not images:
        raise HTTPException(status_code=400, detail="No valid images found in batch.")

    # Vectorized batch prediction
    predictions = classifier.predict_batch(images)
    
    records = []
    for meta, pred in zip(file_metadata, predictions):
        r = {**meta, **pred}
        records.append(r)

    # Bulk insert
    inserted, duplicates = bulk_insert_tiles(records)
    total_time_ms = round((time.perf_counter() - start_all) * 1000.0, 2)
    avg_latency = round(total_time_ms / len(images), 2)

    return BatchClassificationResponse(
        batch_size=len(images),
        inserted_count=inserted,
        duplicate_count=duplicates,
        total_latency_ms=total_time_ms,
        avg_latency_ms=avg_latency,
        runtime_engine=classifier.runtime_engine,
        results=records
    )

@app.post(
    f"{settings.API_PREFIX}/tiles/{{tile_id}}/correct",
    response_model=ClassificationResult,
    tags=["Human-in-the-Loop Active Learning"]
)
def correct_tile_label(tile_id: str, correction: CorrectionRequest):
    """
    Enables analysts to override ambiguous predictions, creating an offline active learning audit log.
    """
    if correction.verified_label not in settings.CLASSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid class. Must be one of: {settings.CLASSES}"
        )
    updated = update_tile_verification(
        tile_id=tile_id,
        verified_label=correction.verified_label,
        notes=correction.analyst_notes
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tile '{tile_id}' not found."
        )
    return updated

@app.get(
    f"{settings.API_PREFIX}/tiles",
    response_model=QueryResponse,
    tags=["Analyst Query Interface"]
)
def get_tiles(
    label: Optional[str] = Query(None, description="Filter by land-use class"),
    status: Optional[str] = Query(None, description="Filter by status: 'CONFIDENT', 'UNCERTAIN', 'VERIFIED'"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    max_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    max_entropy: Optional[float] = Query(None, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    records, total = query_tiles(
        label=label,
        status=status,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        max_entropy=max_entropy,
        limit=limit,
        offset=offset
    )
    return QueryResponse(
        total=total,
        limit=limit,
        offset=offset,
        results=records
    )

@app.get(
    f"{settings.API_PREFIX}/tiles/{{tile_id}}",
    response_model=ClassificationResult,
    tags=["Analyst Query Interface"]
)
def get_single_tile(tile_id: str):
    record = get_tile_by_id(tile_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tile with id '{tile_id}' not found."
        )
    return record

@app.get(
    f"{settings.API_PREFIX}/tiles/{{tile_id}}/image",
    tags=["Analyst Query Interface"]
)
def get_tile_image(tile_id: str):
    """
    Streams raw tile binary to the browser/dashboard.
    """
    record = get_tile_by_id(tile_id)
    if not record or not os.path.exists(record["stored_path"]):
        raise HTTPException(status_code=404, detail="Tile image not found on disk.")
    return FileResponse(record["stored_path"])
