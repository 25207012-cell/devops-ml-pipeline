import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))


@pytest.fixture(scope="session", autouse=True)
def trained_model(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    model_path = d / "model.joblib"
    env = {**os.environ, "MODEL_PATH": str(model_path),
           "MLFLOW_TRACKING_URI": f"file:{d / 'mlruns'}"}
    subprocess.check_call([sys.executable, os.path.join(ROOT, "app", "train.py")], env=env)
    os.environ["MODEL_PATH"] = str(model_path)
    os.environ.pop("MLFLOW_TRACKING_URI", None)


@pytest.fixture()
def client():
    import importlib
    import main
    importlib.reload(main)
    return main.app.test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json["status"] == "ok"


def test_predict_setosa(client):
    r = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert r.status_code == 200
    assert r.json["prediction"] == "setosa"


def test_predict_bad_input(client):
    r = client.post("/predict", json={"features": [1, 2]})
    assert r.status_code == 400


def test_metrics_exposed(client):
    client.post("/predict", json={"features": [6.3, 3.3, 6.0, 2.5]})
    body = client.get("/metrics").get_data(as_text=True)
    assert "app_requests_total" in body
    assert "app_predictions_total" in body
