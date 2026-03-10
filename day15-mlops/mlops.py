"""
Day 15: MLOps basics — model versioning, data drift detection, simple serving.

Covers:
1. Model training + saving/loading (joblib)
2. Data drift detection (comparing distributions)
3. Simple model serving (FastAPI endpoint)
4. Prediction logging for monitoring

Usage:
    python mlops.py train       # Train and save model
    python mlops.py predict     # Load model and predict
    python mlops.py drift       # Detect data drift
    python mlops.py serve       # Start FastAPI server (requires uvicorn)
"""

import os
import sys
import json
import time
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats


MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
LOG_PATH = MODEL_DIR / "predictions.jsonl"


def section(title: str):
    """Print a section header."""
    title = str(title).strip() or "Untitled"
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}\n")


# ========== 1. TRAIN + SAVE ==========

def train_and_save():
    section("Training model")
    MODEL_DIR.mkdir(exist_ok=True)

    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3,
                                random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                          random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save model
    joblib.dump(model, MODEL_PATH)

    # Save metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "version": "1.0.0",
        "trained_at": datetime.now().isoformat(),
        "n_features": 5,
        "n_samples_train": len(X_train),
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "training_data_stats": {
            "means": X_train.mean(axis=0).round(4).tolist(),
            "stds": X_train.std(axis=0).round(4).tolist(),
        },
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved: {MODEL_PATH}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(f"Metadata: {METADATA_PATH}")

    return model, X_train


# ========== 2. LOAD + PREDICT ==========

def load_and_predict():
    """Load a saved model and run sample predictions."""
    section("Loading model and predicting")

    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        print("Model or metadata missing. Run: python mlops.py train")
        return

    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    print(f"Loaded model v{metadata['version']} (trained {metadata['trained_at']})")
    print(f"Training accuracy: {metadata['accuracy']}")

    # Generate some test inputs
    np.random.seed(99)
    X_new = np.random.randn(5, metadata["n_features"])
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)

    for i in range(len(X_new)):
        print(f"\n  Input {i}: {X_new[i].round(3)}")
        print(f"  Prediction: {predictions[i]}")
        print(f"  Confidence: {probabilities[i].max():.3f}")

        # Log prediction
        log_prediction(X_new[i], predictions[i], probabilities[i].max())


def log_prediction(features, prediction, confidence):
    """Append prediction to JSONL log for monitoring."""
    MODEL_DIR.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "features": features.tolist(),
        "prediction": int(prediction),
        "confidence": round(float(confidence), 4),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ========== 3. DATA DRIFT DETECTION ==========

def detect_drift():
    """Compare reference stats against simulated production data."""
    section("Data Drift Detection")

    if not METADATA_PATH.exists():
        print("No model metadata found. Run: python mlops.py train")
        return

    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    stats_data = metadata.get("training_data_stats", {})
    train_means = np.array(stats_data.get("means", []))
    train_stds = np.array(stats_data.get("stds", []))
    if len(train_means) != 5 or len(train_stds) != 5:
        print("Invalid training stats in metadata")
        return

    # Simulate "production" data with drift
    np.random.seed(123)

    scenarios = {
        "No drift (same distribution)": np.random.randn(200, 5),
        "Mild drift (shifted mean)": np.random.randn(200, 5) + 0.5,
        "Severe drift (different distribution)": np.random.randn(200, 5) * 3 + 2,
    }

    for name, X_prod in scenarios.items():
        print(f"\n  Scenario: {name}")
        drift_detected = False

        for feature_idx in range(X_prod.shape[1]):
            # Generate reference data from training stats
            ref_data = np.random.normal(
                train_means[feature_idx],
                train_stds[feature_idx],
                size=200,
            )

            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(ref_data, X_prod[:, feature_idx])

            if p_value < 0.05:
                drift_detected = True
                print(f"    Feature {feature_idx}: DRIFT (p={p_value:.4f}, KS={ks_stat:.3f})")

        if not drift_detected:
            print(f"    No drift detected (all p-values > 0.05)")


# ========== 4. SIMPLE API SERVER ==========

def serve():
    section("Model Serving (FastAPI)")

    try:
        from fastapi import FastAPI
        import uvicorn
    except ImportError:
        print("FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn")
        return

    if not MODEL_PATH.exists():
        print("No model found. Run: python mlops.py train")
        return

    app = FastAPI(title="ML Model Server")
    model = joblib.load(MODEL_PATH)

    @app.post("/predict")
    def predict(features: list[float]):
        if not features or len(features) != getattr(model, "n_features_in_", len(features)):
            return {"error": f"expected {getattr(model, 'n_features_in_', 0)} features"}
        X = np.array(features, dtype=float).reshape(1, -1)
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0].max()
        log_prediction(X[0], pred, prob)
        return {"prediction": int(pred), "confidence": round(float(prob), 4)}

    @app.get("/health")
    def health():
        return {"status": "ok", "model": str(MODEL_PATH)}

    print("Starting server on http://localhost:8000")
    print("POST /predict — send {\"features\": [1.0, 2.0, 3.0, 4.0, 5.0]}")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"

    if cmd == "train":
        train_and_save()
    elif cmd == "predict":
        load_and_predict()
    elif cmd == "drift":
        detect_drift()
    elif cmd == "serve":
        serve()
    else:
        print(f"Unknown command: {cmd}. Use: train, predict, drift, serve")
