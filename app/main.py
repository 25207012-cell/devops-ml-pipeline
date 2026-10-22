"""Iris classifier service with Prometheus metrics.

Loads the model registered in MLflow (models:/iris-classifier/Production-or-latest).
Falls back to a local model.joblib so the container still starts without MLflow
(useful for CI and Q1 where only Docker Compose is required).
"""
import os
import time

import joblib
from flask import Flask, jsonify, request
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)

app = Flask(__name__)

REQUESTS = Counter("app_requests_total", "Total HTTP requests", ["endpoint", "status"])
LATENCY = Histogram("app_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTIONS = Counter("app_predictions_total", "Predictions by class", ["label"])

CLASS_NAMES = ["setosa", "versicolor", "virginica"]
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_URI = os.getenv("MODEL_URI", "models:/iris-classifier/latest")


def load_model():
    if MLFLOW_URI:
        try:
            import mlflow.sklearn
            mlflow.set_tracking_uri(MLFLOW_URI)
            return mlflow.sklearn.load_model(MODEL_URI), f"mlflow:{MODEL_URI}"
        except Exception as exc:  # fall back to local file
            print(f"MLflow load failed ({exc}); using local model", flush=True)
    return joblib.load(MODEL_PATH), f"file:{MODEL_PATH}"


model, model_source = load_model()


@app.get("/")
def index():
    return jsonify(service="iris-classifier", model_source=model_source)


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    start = time.time()
    try:
        features = request.get_json(force=True)["features"]
        if len(features) != 4:
            raise ValueError("features must have 4 values")
        idx = int(model.predict([features])[0])
        label = CLASS_NAMES[idx]
        PREDICTIONS.labels(label).inc()
        REQUESTS.labels("/predict", "200").inc()
        return jsonify(prediction=label)
    except Exception as exc:
        REQUESTS.labels("/predict", "400").inc()
        return jsonify(error=str(exc)), 400
    finally:
        LATENCY.labels("/predict").observe(time.time() - start)


@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
