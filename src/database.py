"""
Tier 2: Relational Metadata & Inference Log Database (SQLite)
Provides persistent storage for UAV segmentation inference history,
field coverage analytics, infestation risk levels, and agronomic decisions.
"""

import os
import sys
import sqlite3
from datetime import datetime
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "outputs/agrifusion.db")

def get_connection(db_path=None):
    db_path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    return conn

def init_db(db_path=None):
    """Initializes the SQLite database and creates the predictions table."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_name TEXT NOT NULL,
            weed_coverage_pct REAL NOT NULL,
            crop_coverage_pct REAL NOT NULL,
            weed_to_crop_ratio REAL,
            physical_weed_area_m2 REAL,
            physical_crop_area_m2 REAL,
            high_risk_zones INTEGER,
            risk_level TEXT,
            recommended_action TEXT,
            inference_time_ms REAL
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(
    image_name: str,
    coverage: dict,
    heatmap_summary: dict = None,
    decision_report: dict = None,
    inference_time_ms: float = 0.0,
    db_path=None
):
    """
    Inserts an inference result into the database.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    weed_pct = coverage.get("weed_coverage_pct", 0.0)
    crop_pct = coverage.get("crop_coverage_pct", 0.0)
    ratio = coverage.get("weed_to_crop_ratio", 0.0)
    weed_m2 = coverage.get("physical_weed_area_m2", 0.0)
    crop_m2 = coverage.get("physical_crop_area_m2", 0.0)
    
    high_zones = heatmap_summary.get("high_infestation_zones", 0) if heatmap_summary else 0
    risk = decision_report.get("assessment_priority", "UNKNOWN") if decision_report else "N/A"
    action = decision_report.get("recommended_action", "N/A") if decision_report else "N/A"
    
    cursor.execute("""
        INSERT INTO predictions (
            timestamp, image_name, weed_coverage_pct, crop_coverage_pct,
            weed_to_crop_ratio, physical_weed_area_m2, physical_crop_area_m2,
            high_risk_zones, risk_level, recommended_action, inference_time_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp, os.path.basename(image_name), weed_pct, crop_pct,
        ratio, weed_m2, crop_m2, high_zones, risk, action, round(inference_time_ms, 2)
    ))
    
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def get_prediction_history(limit: int = 100, db_path=None) -> pd.DataFrame:
    """Retrieves recent inference logs as a pandas DataFrame."""
    init_db(db_path)
    conn = get_connection(db_path)
    query = f"SELECT * FROM predictions ORDER BY id DESC LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def clear_prediction_history(db_path=None):
    """Clears all records in the predictions table."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print(f"[PASS] SQLite database initialized at: {DEFAULT_DB_PATH}")
    df = get_prediction_history()
    print(f"Current logged records: {len(df)}")

