import os
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "GalaxEye Offline Satellite Tile Classifier & ML Systems Backend"
    VERSION: str = "2.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Paths
    BASE_DIR: str = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    MODEL_DIR: str = os.path.join(BASE_DIR, "models")
    MODEL_PATH: str = os.path.join(MODEL_DIR, "classifier.pt")
    ONNX_MODEL_PATH: str = os.path.join(MODEL_DIR, "classifier.onnx")
    METADATA_PATH: str = os.path.join(MODEL_DIR, "model_metadata.json")
    
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    DB_PATH: str = os.path.join(DATA_DIR, "tiles_catalog.db")
    STORAGE_DIR: str = os.path.join(DATA_DIR, "storage")
    
    # ML & System Thresholds
    # Multi-factor Uncertainty Gate:
    CONFIDENCE_THRESHOLD: float = 0.60
    MARGIN_THRESHOLD: float = 0.20       # Difference between top-1 and top-2 class probability
    MAX_ENTROPY_THRESHOLD: float = 0.65   # Normalized Shannon Entropy [0, 1]
    
    # Supported classes
    CLASSES: list[str] = [
        "AnnualCrop", "Forest", "Highway", "Industrial", 
        "Residential", "River", "SeaLake"
    ]
    
    # Image constraints
    IMAGE_SIZE: tuple[int, int] = (64, 64)
    ALLOWED_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

settings = Settings()

# Ensure directories exist
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.STORAGE_DIR, exist_ok=True)
os.makedirs(settings.MODEL_DIR, exist_ok=True)
