from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class ClassificationResult(BaseModel):
    id: str
    sha256_hash: str
    original_filename: str
    stored_path: str
    file_size_bytes: int
    image_width: int
    image_height: int
    image_channels: int
    predicted_class: str
    confidence: float
    margin: float = 0.0
    entropy: float = 0.0
    class_probabilities: Dict[str, float]
    status: str  # 'CONFIDENT', 'UNCERTAIN', 'VERIFIED'
    verified_label: Optional[str] = None
    analyst_notes: Optional[str] = None
    model_version: str
    inference_latency_ms: float
    created_at: str
    is_duplicate: Optional[bool] = False

class QueryResponse(BaseModel):
    total: int
    limit: int
    offset: int
    results: List[ClassificationResult]

class BatchClassificationResponse(BaseModel):
    batch_size: int
    inserted_count: int
    duplicate_count: int
    total_latency_ms: float
    avg_latency_ms: float
    runtime_engine: str
    results: List[ClassificationResult]

class CorrectionRequest(BaseModel):
    verified_label: str
    analyst_notes: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    offline_mode: bool
    runtime_engine: str
    model_version: str
    model_loaded: bool
    database_connected: bool
    total_tiles_processed: int
    timestamp: str

class SystemStatsResponse(BaseModel):
    total_tiles: int
    status_distribution: Dict[str, int]
    class_distribution: Dict[str, int]
    avg_confidence: float
    avg_entropy: float
    avg_latency_ms: float
    human_corrections_recorded: int
