import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from app.config import settings

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Core tiles table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tiles (
        id TEXT PRIMARY KEY,
        sha256_hash TEXT UNIQUE NOT NULL,
        original_filename TEXT NOT NULL,
        stored_path TEXT NOT NULL,
        file_size_bytes INTEGER NOT NULL,
        image_width INTEGER NOT NULL,
        image_height INTEGER NOT NULL,
        image_channels INTEGER NOT NULL,
        predicted_class TEXT NOT NULL,
        confidence REAL NOT NULL,
        margin REAL NOT NULL DEFAULT 0.0,
        entropy REAL NOT NULL DEFAULT 0.0,
        class_probabilities TEXT NOT NULL,
        status TEXT NOT NULL,  -- 'CONFIDENT', 'UNCERTAIN', 'VERIFIED'
        verified_label TEXT,
        analyst_notes TEXT,
        model_version TEXT NOT NULL,
        inference_latency_ms REAL NOT NULL,
        created_at TEXT NOT NULL
    );
    """)
    
    # Audit log / active learning corrections table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_corrections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tile_id TEXT NOT NULL,
        original_prediction TEXT NOT NULL,
        verified_label TEXT NOT NULL,
        original_confidence REAL NOT NULL,
        analyst_notes TEXT,
        corrected_at TEXT NOT NULL,
        FOREIGN KEY (tile_id) REFERENCES tiles (id)
    );
    """)
    
    # Ensure column migrations for existing databases
    cursor.execute("PRAGMA table_info(tiles);")
    existing_cols = {row[1] for row in cursor.fetchall()}
    if "entropy" not in existing_cols:
        cursor.execute("ALTER TABLE tiles ADD COLUMN entropy REAL NOT NULL DEFAULT 0.0;")
    if "margin" not in existing_cols:
        cursor.execute("ALTER TABLE tiles ADD COLUMN margin REAL NOT NULL DEFAULT 0.0;")
    if "verified_label" not in existing_cols:
        cursor.execute("ALTER TABLE tiles ADD COLUMN verified_label TEXT;")
    if "analyst_notes" not in existing_cols:
        cursor.execute("ALTER TABLE tiles ADD COLUMN analyst_notes TEXT;")

    # Indexes for rapid analyst querying
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tiles_predicted_class ON tiles (predicted_class);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tiles_status ON tiles (status);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tiles_confidence ON tiles (confidence);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tiles_entropy ON tiles (entropy);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tiles_created_at ON tiles (created_at);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tiles_sha256 ON tiles (sha256_hash);")
    
    conn.commit()
    conn.close()

def insert_tile_record(record: Dict[str, Any]) -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM tiles WHERE sha256_hash = ?", (record["sha256_hash"],))
    existing = cursor.fetchone()
    if existing:
        conn.close()
        res = dict(existing)
        res["class_probabilities"] = json.loads(res["class_probabilities"])
        res["is_duplicate"] = True
        return res
    
    cursor.execute("""
    INSERT INTO tiles (
        id, sha256_hash, original_filename, stored_path, file_size_bytes,
        image_width, image_height, image_channels, predicted_class,
        confidence, margin, entropy, class_probabilities, status,
        verified_label, analyst_notes, model_version,
        inference_latency_ms, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record["id"],
        record["sha256_hash"],
        record["original_filename"],
        record["stored_path"],
        record["file_size_bytes"],
        record["image_width"],
        record["image_height"],
        record["image_channels"],
        record["predicted_class"],
        record["confidence"],
        record.get("margin", 0.0),
        record.get("entropy", 0.0),
        json.dumps(record["class_probabilities"]),
        record["status"],
        record.get("verified_label"),
        record.get("analyst_notes"),
        record["model_version"],
        record["inference_latency_ms"],
        record.get("created_at", datetime.utcnow().isoformat())
    ))
    conn.commit()
    conn.close()
    
    record["is_duplicate"] = False
    return record

def bulk_insert_tiles(records: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    High-throughput bulk insert using a single transaction.
    Returns (inserted_count, duplicate_count).
    """
    if not records:
        return 0, 0
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    inserted = 0
    duplicates = 0
    
    cursor.execute("BEGIN TRANSACTION;")
    try:
        for r in records:
            cursor.execute("SELECT id FROM tiles WHERE sha256_hash = ?", (r["sha256_hash"],))
            if cursor.fetchone():
                duplicates += 1
                continue
                
            cursor.execute("""
            INSERT INTO tiles (
                id, sha256_hash, original_filename, stored_path, file_size_bytes,
                image_width, image_height, image_channels, predicted_class,
                confidence, margin, entropy, class_probabilities, status,
                verified_label, analyst_notes, model_version,
                inference_latency_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r["id"],
                r["sha256_hash"],
                r["original_filename"],
                r["stored_path"],
                r["file_size_bytes"],
                r["image_width"],
                r["image_height"],
                r["image_channels"],
                r["predicted_class"],
                r["confidence"],
                r.get("margin", 0.0),
                r.get("entropy", 0.0),
                json.dumps(r["class_probabilities"]),
                r["status"],
                r.get("verified_label"),
                r.get("analyst_notes"),
                r["model_version"],
                r["inference_latency_ms"],
                r.get("created_at", datetime.utcnow().isoformat())
            ))
            inserted += 1
        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.close()
        raise e
        
    conn.close()
    return inserted, duplicates

def update_tile_verification(tile_id: str, verified_label: str, notes: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Records an analyst's verified ground truth correction (Human-in-the-Loop active learning).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM tiles WHERE id = ?", (tile_id,))
    tile = cursor.fetchone()
    if not tile:
        conn.close()
        return None
        
    now = datetime.utcnow().isoformat()
    
    # Update tile record
    cursor.execute("""
    UPDATE tiles 
    SET status = 'VERIFIED', verified_label = ?, analyst_notes = ?
    WHERE id = ?
    """, (verified_label, notes, tile_id))
    
    # Insert audit record
    cursor.execute("""
    INSERT INTO audit_corrections (
        tile_id, original_prediction, verified_label, original_confidence,
        analyst_notes, corrected_at
    ) VALUES (?, ?, ?, ?, ?, ?)
    """, (tile_id, tile["predicted_class"], verified_label, tile["confidence"], notes, now))
    
    conn.commit()
    
    cursor.execute("SELECT * FROM tiles WHERE id = ?", (tile_id,))
    updated = cursor.fetchone()
    conn.close()
    
    res = dict(updated)
    res["class_probabilities"] = json.loads(res["class_probabilities"])
    return res

def query_tiles(
    label: Optional[str] = None,
    status: Optional[str] = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
    max_entropy: Optional[float] = None,
    limit: int = 50,
    offset: int = 0
) -> Tuple[List[Dict[str, Any]], int]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    filters = []
    params = []
    
    if label:
        filters.append("(predicted_class = ? OR verified_label = ?)")
        params.extend([label, label])
    if status:
        filters.append("status = ?")
        params.append(status.upper())
    if min_confidence is not None:
        filters.append("confidence >= ?")
        params.append(min_confidence)
    if max_confidence is not None:
        filters.append("confidence <= ?")
        params.append(max_confidence)
    if max_entropy is not None:
        filters.append("entropy <= ?")
        params.append(max_entropy)
        
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    
    count_query = f"SELECT COUNT(*) FROM tiles {where_clause}"
    cursor.execute(count_query, params)
    total_count = cursor.fetchone()[0]
    
    query = f"""
    SELECT * FROM tiles
    {where_clause}
    ORDER BY created_at DESC
    LIMIT ? OFFSET ?
    """
    cursor.execute(query, params + [limit, offset])
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        d = dict(r)
        d["class_probabilities"] = json.loads(d["class_probabilities"])
        results.append(d)
        
    return results, total_count

def get_tile_by_id(tile_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tiles WHERE id = ?", (tile_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["class_probabilities"] = json.loads(d["class_probabilities"])
    return d

def get_system_stats() -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM tiles")
    total_tiles = cursor.fetchone()[0]
    
    cursor.execute("SELECT status, COUNT(*) FROM tiles GROUP BY status")
    status_counts = dict(cursor.fetchall())
    
    cursor.execute("SELECT predicted_class, COUNT(*) FROM tiles GROUP BY predicted_class")
    class_counts = dict(cursor.fetchall())
    
    cursor.execute("SELECT AVG(confidence), AVG(entropy), AVG(inference_latency_ms) FROM tiles")
    avg_conf, avg_entropy, avg_latency = cursor.fetchone()

    cursor.execute("SELECT COUNT(*) FROM audit_corrections")
    corrections_count = cursor.fetchone()[0]
    
    conn.close()
    return {
        "total_tiles": total_tiles,
        "status_distribution": status_counts,
        "class_distribution": class_counts,
        "avg_confidence": round(avg_conf, 4) if avg_conf else 0.0,
        "avg_entropy": round(avg_entropy, 4) if avg_entropy else 0.0,
        "avg_latency_ms": round(avg_latency, 2) if avg_latency else 0.0,
        "human_corrections_recorded": corrections_count
    }
