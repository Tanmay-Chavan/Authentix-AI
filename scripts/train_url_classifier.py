"""Training script for the URL phishing detection model (XGBoost).

This model trains on lexical URL features — no neural network needed.
XGBoost is fast, interpretable, and perfect for tabular feature data.

Training time: ~2-5 minutes on CPU for 650K URLs.

Usage:
    python -m scripts.train_url_classifier --max-samples 10000
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import GradientBoostingClassifier

import structlog

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from authentix.pipelines.data_registry import DataRegistry
from authentix.pipelines.url.feature_extractor import FEATURE_NAMES

logger = structlog.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train Authentix URL phishing classifier"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples to use (for quick testing)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Test split ratio",
    )
    parser.add_argument(
        "--data-dir", type=str, default="d:/Files/DMCE/Authentix/data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models_store",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=200,
        help="Number of boosting rounds",
    )
    parser.add_argument(
        "--max-depth", type=int, default=6,
        help="Max tree depth",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.1,
        help="Boosting learning rate",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Data ──
    logger.info("url_training.loading_data", max_samples=args.max_samples)
    registry = DataRegistry(args.data_dir)
    url_ds = registry.get_url_dataset(max_samples=args.max_samples)
    X, y = url_ds.get_features_and_labels()

    logger.info(
        "url_training.data_loaded",
        samples=len(y),
        features=X.shape[1],
        benign=int(sum(y == 0)),
        malicious=int(sum(y == 1)),
    )

    # ── Train/Test Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    logger.info(
        "url_training.split",
        train=len(y_train),
        test=len(y_test),
    )

    # ── Train Gradient Boosting ──
    # Using sklearn's GradientBoostingClassifier (no XGBoost dependency needed)
    # For production, can swap in XGBoost for better performance
    logger.info("url_training.training_start")
    start_time = time.time()

    class_weights = url_ds.class_weights
    # Convert class weights to sample weights
    sample_weights = np.array([class_weights[label] for label in y_train])

    model = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        verbose=1,
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)
    train_time = time.time() - start_time

    # ── Evaluate ──
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_time": f"{train_time:.1f}s",
        "n_samples": len(y),
        "n_features": int(X.shape[1]),
    }

    # ── Feature Importance ──
    importances = model.feature_importances_
    feature_importance = sorted(
        zip(FEATURE_NAMES, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    # ── Save Model ──
    model_path = output_dir / "url_classifier_best.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save metrics
    metrics_path = output_dir / "url_classifier_history.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "feature_importance": [
                {"feature": name, "importance": float(imp)}
                for name, imp in feature_importance
            ],
        }, f, indent=2)

    # ── Print Results ──
    print("\n" + "=" * 60)
    print("URL CLASSIFIER TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Samples:       {len(y):,}")
    print(f"  Features:      {X.shape[1]}")
    print(f"  Train Time:    {train_time:.1f}s")
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print(f"  F1 Score:      {metrics['f1']:.4f}")
    print(f"  AUC-ROC:       {metrics['auc_roc']:.4f}")
    print(f"\n  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"    TN={cm[0][0]:,}  FP={cm[0][1]:,}")
    print(f"    FN={cm[1][0]:,}  TP={cm[1][1]:,}")
    print(f"\n  Top 10 Features:")
    for name, imp in feature_importance[:10]:
        print(f"    {name:30s} {imp:.4f}")
    print(f"\n  Model saved:   {model_path}")
    print("=" * 60)

    print("\n" + classification_report(y_test, y_pred, target_names=["Benign", "Malicious"]))


if __name__ == "__main__":
    main()
