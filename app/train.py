"""Train an Iris classifier and track everything with MLflow.

Logs params + metrics, registers each run as a new version of the model
"iris-classifier", and also writes model.joblib for the Docker image.
Run several times with different --n-estimators / --max-depth to get
multiple experiment runs and model versions.
"""
import argparse
import os

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--register", action="store_true", help="register model in MLflow registry")
    args = p.parse_args()

    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("iris-experiment")

    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
        ).fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, average="macro")

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        kwargs = {}
        if args.register:
            kwargs["registered_model_name"] = "iris-classifier"
        mlflow.sklearn.log_model(clf, "model", **kwargs)

        joblib.dump(clf, os.getenv("MODEL_PATH", "model.joblib"))
        print(f"run_id={run.info.run_id} accuracy={acc:.4f} f1_macro={f1:.4f}")


if __name__ == "__main__":
    main()
