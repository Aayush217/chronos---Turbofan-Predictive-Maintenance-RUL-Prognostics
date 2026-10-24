"""
Chronos: Training and Deployment Pipeline
Executes end-to-end training, cross-validation, test-set evaluation, and artifact serialization.
"""

import os
import json
import numpy as np
import pandas as pd

from .data_loader import load_cmapss_train, load_cmapss_test
from .feature_engineering import TelemetryFeaturePipeline
from .models import RULRegressionModel, FailureClassificationModel, save_model
from .evaluation import evaluate_rul_regression, evaluate_risk_classification

def run_training_pipeline(dataset_dir: str, output_models_dir: str, subset: str = "FD001"):
    print("=" * 65)
    print(f"  Chronos: End-to-End Predictive Maintenance Pipeline")
    print(f"  Dataset: NASA C-MAPSS ({subset})")
    print("=" * 65)
    
    # 1. Ingestion
    print("\n[1/5] Ingesting telemetry data...")
    train_df = load_cmapss_train(dataset_dir, subset=subset, max_rul_clip=125)
    test_df, true_test_rul = load_cmapss_test(dataset_dir, subset=subset)
    print(f"  -> Training records: {len(train_df)} across {train_df['engine_id'].nunique()} engines")
    print(f"  -> Test records: {len(test_df)} across {test_df['engine_id'].nunique()} test engines")

    # 2. Feature Engineering
    print("\n[2/5] Engineering temporal features & rolling degradation statistics...")
    pipeline = TelemetryFeaturePipeline(window_size=5)
    X_train = pipeline.fit_transform(train_df)
    y_train_rul = train_df["RUL_clipped"].values
    y_train_clf = train_df["failure_soon"].values
    
    # Extract test features at the final operational cycle of each test engine
    X_test_last = pipeline.extract_test_latest_features(test_df)
    y_test_rul = true_test_rul.values
    y_test_clf = (y_test_rul <= 30).astype(int)
    print(f"  -> Engineered features count: {X_train.shape[1]}")

    # 3. Model Training
    print("\n[3/5] Training dual-task AI models...")
    print("  -> Training RUL Regressor (Random Forest Regressor)...")
    regressor = RULRegressionModel(model_type="random_forest", n_estimators=100, max_depth=14)
    regressor.fit(X_train, y_train_rul)

    print("  -> Training Failure Risk Classifier (Balanced Random Forest)...")
    classifier = FailureClassificationModel(model_type="random_forest", n_estimators=100, max_depth=12)
    classifier.fit(X_train, y_train_clf)

    # 4. Out-of-Sample Test Evaluation
    print("\n[4/5] Evaluating performance on 100 blind test engines...")
    # Regression predictions
    pred_test_rul = regressor.predict(X_test_last)
    reg_metrics = evaluate_rul_regression(y_test_rul, pred_test_rul)
    
    # Classification predictions
    pred_test_clf = classifier.predict(X_test_last)
    pred_test_prob = classifier.predict_proba(X_test_last)[:, 1]
    clf_metrics = evaluate_risk_classification(y_test_clf, pred_test_clf, pred_test_prob)

    print("\n  ================ RUL REGRESSION METRICS ================")
    print(f"  RMSE:        {reg_metrics['RMSE']:.2f} cycles")
    print(f"  MAE:         {reg_metrics['MAE']:.2f} cycles")
    print(f"  R2 Score:    {reg_metrics['R2_Score']:.4f}")
    print(f"  NASA Score:  {reg_metrics['NASA_Score']:.2f}")

    print("\n  ============ 30-CYCLE RISK CLASSIFICATION =============")
    print(f"  Accuracy:    {clf_metrics['Accuracy'] * 100:.2f}%")
    print(f"  Precision:   {clf_metrics['Precision'] * 100:.2f}%")
    print(f"  Recall:      {clf_metrics['Recall'] * 100:.2f}%")
    print(f"  F1-Score:    {clf_metrics['F1_Score']:.4f}")
    print(f"  ROC-AUC:     {clf_metrics['ROC_AUC']:.4f}")
    print("  ========================================================")

    # 5. Serialization
    print("\n[5/5] Saving production model artifacts...")
    os.makedirs(output_models_dir, exist_ok=True)
    
    reg_path = os.path.join(output_models_dir, "rul_regressor.joblib")
    clf_path = os.path.join(output_models_dir, "risk_classifier.joblib")
    pipeline_path = os.path.join(output_models_dir, "feature_pipeline.joblib")
    meta_path = os.path.join(output_models_dir, "metadata.json")

    save_model(regressor, reg_path)
    save_model(classifier, clf_path)
    save_model(pipeline, pipeline_path)

    metadata = {
        "project": "Chronos",
        "dataset_subset": subset,
        "features_count": X_train.shape[1],
        "regression_metrics": reg_metrics,
        "classification_metrics": clf_metrics,
        "feature_names": pipeline.feature_names
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"  -> Model artifacts successfully stored in: {output_models_dir}")
    print("\nPipeline execution completed successfully.")
    return metadata

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    dataset_dir = os.path.join(project_root, "dataset", "CMAPSSData")
    models_dir = os.path.join(project_root, "models")
    
    run_training_pipeline(dataset_dir, models_dir, subset="FD001")
