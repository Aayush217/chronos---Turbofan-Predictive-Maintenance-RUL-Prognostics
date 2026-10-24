"""
Chronos: Interactive Engine Diagnostics & Prediction CLI
Interactive demonstration tool for live project presentations and health monitoring.
Usage:
    python predict_engine.py --engine 24
    python predict_engine.py --batch
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models import load_model
from src.data_loader import load_cmapss_test, INFORMATIVE_SENSORS

def assess_health_status(predicted_rul: float, failure_prob: float):
    if predicted_rul <= 30 or failure_prob >= 0.70:
        return "CRITICAL [ALERT]", "RED", "Ground engine immediately. Schedule urgent level-3 turbine overhaul."
    elif predicted_rul <= 75 or failure_prob >= 0.35:
        return "CAUTION [MONITOR]", "YELLOW", "Schedule inspection within 15 cycles. Monitor HPC pressure and temperature."
    else:
        return "NOMINAL [HEALTHY]", "GREEN", "Engine operating within normal safety envelopes. No maintenance required."

def inspect_engine(engine_id: int, project_root: str, save_plot: bool = True):
    dataset_dir = os.path.join(project_root, "dataset", "CMAPSSData")
    models_dir = os.path.join(project_root, "models")
    
    # 1. Load artifacts
    pipeline = load_model(os.path.join(models_dir, "feature_pipeline.joblib"))
    regressor = load_model(os.path.join(models_dir, "rul_regressor.joblib"))
    classifier = load_model(os.path.join(models_dir, "risk_classifier.joblib"))
    
    # 2. Ingest test engine telemetry
    test_df, true_rul_series = load_cmapss_test(dataset_dir, subset="FD001")
    
    if engine_id not in test_df["engine_id"].unique():
        raise ValueError(f"Engine ID {engine_id} not found in test dataset (Valid: 1 to 100).")
        
    engine_telemetry = test_df[test_df["engine_id"] == engine_id].copy()
    actual_rul = true_rul_series.iloc[engine_id - 1]
    last_cycle = engine_telemetry["cycle"].max()
    
    # 3. Transform features and predict
    transformed_features = pipeline.transform(engine_telemetry)
    latest_feature_vector = transformed_features[-1].reshape(1, -1)
    
    pred_rul = float(regressor.predict(latest_feature_vector)[0])
    pred_risk_prob = float(classifier.predict_proba(latest_feature_vector)[0, 1])
    
    status_label, color, action_plan = assess_health_status(pred_rul, pred_risk_prob)
    
    # Print Diagnostic Report
    print("\n" + "=" * 70)
    print(f"       CHRONOS : TURBOFAN DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"  Target Unit:                Engine #{engine_id}")
    print(f"  Recorded Flight Cycles:     {last_cycle} cycles")
    print(f"  Estimated Remaining Life:   {pred_rul:.1f} cycles")
    print(f"  True Ground-Truth RUL:      {actual_rul:.1f} cycles")
    print(f"  Prediction Error:           {abs(pred_rul - actual_rul):.1f} cycles ({(pred_rul - actual_rul):+.1f})")
    print(f"  Failure Probability (<=30c): {pred_risk_prob * 100:.1f}%")
    print(f"  Operational Health Status:  {status_label}")
    print(f"  Prescribed Maintenance:     {action_plan}")
    print("=" * 70)
    
    # 4. Telemetry Plot
    if save_plot:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(f"Chronos: Real-Time Telemetry Drift - Engine #{engine_id} (Status: {status_label})", fontsize=14, weight='bold')
        
        sensors_to_plot = [("sensor_2", "LPC Outlet Temp (R)"),
                           ("sensor_3", "HPC Outlet Temp (R)"),
                           ("sensor_4", "LPT Outlet Temp (R)"),
                           ("sensor_11", "HPC Static Pressure (psia)")]
        
        for ax, (s_col, s_label) in zip(axes.flatten(), sensors_to_plot):
            ax.plot(engine_telemetry["cycle"], engine_telemetry[s_col], color='navy', lw=2, marker='.')
            ax.set_xlabel("Flight Cycles")
            ax.set_ylabel(s_label)
            ax.set_title(f"Degradation Profile: {s_col}")
            ax.grid(True, linestyle="--", alpha=0.6)
            
        plt.tight_layout()
        plot_path = os.path.join(project_root, f"engine_{engine_id}_telemetry.png")
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"  -> Degradation trajectory chart generated: {plot_path}\n")

def run_batch_evaluation(project_root: str):
    models_dir = os.path.join(project_root, "models")
    meta_path = os.path.join(models_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print("Models not yet trained. Run training pipeline first.")
        return
        
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    print("\n" + "=" * 60)
    print("      CHRONOS BATCH PERFORMANCE BENCHMARK")
    print("=" * 60)
    for category, metrics in [("RUL Regression Metrics", meta["regression_metrics"]), 
                              ("Classification Metrics", meta["classification_metrics"])]:
        print(f"\n  [{category}]")
        for k, v in metrics.items():
            print(f"    - {k:<15}: {v}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chronos Live Inference Engine")
    parser.add_argument("--engine", type=int, default=24, help="Test Engine ID (1 to 100)")
    parser.add_argument("--batch", action="store_true", help="Print overall benchmark summary")
    args = parser.parse_args()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.batch:
        run_batch_evaluation(current_dir)
    else:
        inspect_engine(args.engine, current_dir)
