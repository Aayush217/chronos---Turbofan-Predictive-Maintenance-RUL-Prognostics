"""
Chronos: Data Loader Module
Handles ingestion and RUL calculation for NASA C-MAPSS Turbofan Engine datasets.
"""

import os
import pandas as pd
import numpy as np

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
BASE_COLS = ["engine_id", "cycle"] + SETTING_COLS + SENSOR_COLS

# Sensors with zero or negligible variance across operating cycles in FD001
DROP_SENSORS = ["sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]
INFORMATIVE_SENSORS = [s for s in SENSOR_COLS if s not in DROP_SENSORS]

def load_cmapss_train(dataset_dir: str, subset: str = "FD001", max_rul_clip: int = 125) -> pd.DataFrame:
    """
    Loads training telemetry and calculates true RUL with optional piecewise linear clipping.
    """
    file_path = os.path.join(dataset_dir, f"train_{subset}.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training dataset not found at: {file_path}")
        
    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=BASE_COLS)
    
    # Calculate Remaining Useful Life (RUL)
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    
    # Piecewise-linear RUL clipping (standard industry practice to avoid fitting early-life constant wear)
    if max_rul_clip is not None:
        df["RUL_clipped"] = df["RUL"].clip(upper=max_rul_clip)
    else:
        df["RUL_clipped"] = df["RUL"]
        
    # Binary maintenance alert: failure within 30 cycles
    df["failure_soon"] = (df["RUL"] <= 30).astype(int)
    
    # Multi-class health status
    # 0: Healthy (>90 cycles), 1: Caution (30-90 cycles), 2: Critical (<=30 cycles)
    conditions = [df["RUL"] > 90, (df["RUL"] > 30) & (df["RUL"] <= 90), df["RUL"] <= 30]
    choices = [0, 1, 2]
    df["health_stage"] = np.select(conditions, choices, default=0)
    
    return df

def load_cmapss_test(dataset_dir: str, subset: str = "FD001") -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads test telemetry and ground-truth final RUL vector.
    """
    test_path = os.path.join(dataset_dir, f"test_{subset}.txt")
    rul_path = os.path.join(dataset_dir, f"RUL_{subset}.txt")
    
    if not os.path.exists(test_path) or not os.path.exists(rul_path):
        raise FileNotFoundError(f"Test data or ground-truth RUL missing in: {dataset_dir}")
        
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=BASE_COLS)
    true_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["true_RUL"])["true_RUL"]
    
    return test_df, true_rul
