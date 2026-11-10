"""Регрессии аудита API на синтетическом CSV и временных моделях."""
import csv
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

import main


@pytest.mark.parametrize("classes", [[2, 3], [0, 0], [0], [False, True], [0., 1.]])
def test_incompatible_classes_disable_model(isolated_service, classes):
    with TestClient(isolated_service) as client:
        assert client.post("/model/train").is_success
    artifact = joblib.load(main.MODEL_PATH)
    artifact["pipeline"].named_steps["classifier"].classes_ = np.array(classes)
    joblib.dump(artifact, main.MODEL_PATH)
    with TestClient(isolated_service) as client:
        assert client.get("/model/status").json()["load_error"]
        assert client.get("/health").json()["status"] == "degraded"
        assert client.post("/predict", json=main.PREDICTION_EXAMPLE).status_code == 503


@pytest.mark.parametrize(("key", "value"), [
    ("roc_auc", 7), ("roc_auc", float("nan")), ("roc_auc", None),
    ("accuracy", True), ("f1", float("inf")),
])
def test_invalid_metrics_rejected(client, key, value):
    assert client.post("/model/train").is_success
    artifact = joblib.load(main.MODEL_PATH)
    artifact["metrics"][key] = value
    joblib.dump(artifact, main.MODEL_PATH)
    with pytest.raises(ValueError, match="метрики"):
        main.load_churn_model()


def test_legacy_model_without_auc(isolated_service):
    with TestClient(isolated_service) as client:
        assert client.post("/model/train").is_success
        prediction = client.post("/predict", json=main.PREDICTION_EXAMPLE).json()
    artifact = joblib.load(main.MODEL_PATH)
    del artifact["metrics"]["roc_auc"]
    joblib.dump(artifact, main.MODEL_PATH)
    with TestClient(isolated_service) as client:
        assert client.get("/model/status").json()["metrics"]["roc_auc"] is None
        assert client.get("/model/metrics").json()["latest"]["metrics"]["roc_auc"] is None
        assert client.post("/predict", json=main.PREDICTION_EXAMPLE).json() == prediction


def test_status_waits_for_publication(client, monkeypatch):
    lock = main.training_lock
    reader_waiting = Event()

    class ObservedLock:
        def __enter__(self):
            reader_waiting.set()
            lock.acquire()

        def __exit__(self, *args):
            lock.release()

    assert client.post("/model/train").is_success
    model = main.app.state.churn_model
    monkeypatch.setattr(main, "training_lock", ObservedLock())
    with ThreadPoolExecutor(1) as pool:
        with lock:
            # Воспроизводим окно между двумя присваиваниями первого обучения.
            main.app.state.churn_model = None
            future = pool.submit(client.get, "/model/status")
            assert reader_waiting.wait(3)
            main.app.state.churn_model = model
        status = future.result(timeout=3).json()
    assert status["trained"] and status["model_type"] == "logreg"


@pytest.mark.parametrize(("field", "value"), [
    ("usage_hours", 1e308), ("support_requests", 10**320), ("autopay_enabled", 2),
])
def test_numeric_limits_in_requests_and_csv(client, csv_path, churn_rows, field, value):
    sample = {**main.PREDICTION_EXAMPLE, field: value}
    for body, location in [(sample, ["body", field]), ([sample], ["body", 0, field])]:
        response = client.post("/predict", json=body)
        assert response.status_code == 422
        assert [d["location"] for d in response.json()["details"]] == [location]
    churn_rows[0][field] = value
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=churn_rows[0])
        writer.writeheader()
        writer.writerows(churn_rows)
    response = client.post("/model/train")
    assert response.status_code == 422 and response.json()["code"] == "DATASET_ERROR"


@pytest.mark.parametrize("kind", ["logreg", "random_forest"])
def test_json_class_weights_roundtrip(client, kind):
    weights = {"0": 1, "1": 2}
    result = client.post("/model/train", json={"model_type": kind, "hyperparameters": {"class_weight": weights}})
    assert result.status_code == 200
    assert result.json()["hyperparameters"]["class_weight"] == weights
    assert client.get("/model/status").json()["hyperparameters"]["class_weight"] == weights
    assert main.load_churn_model()["pipeline"].named_steps["classifier"].class_weight == {0: 1, 1: 2}
    invalid = client.post("/model/train", json={"hyperparameters": {"class_weight": {"2": 1}}})
    assert invalid.status_code == 422


@pytest.mark.parametrize("params", ['{"C":1e309}', '{"class_weight":{"0":1,"1":1e309}}', '{"x":[-1e309]}'])
def test_nonfinite_parameters_rejected_before_training(client, params):
    assert client.post("/model/train").is_success
    before = main.MODEL_PATH.read_bytes()
    result = client.post("/model/train", content='{"hyperparameters":' + params + '}',
                         headers={"Content-Type": "application/json"})
    assert result.status_code == 422 and result.json()["code"] == "VALIDATION_ERROR"
    assert main.MODEL_PATH.read_bytes() == before
    assert client.get("/model/metrics").json()["total"] == 1


def test_oversized_header(client, csv_path):
    csv_path.write_text("x" * (csv.field_size_limit() + 1) + "\n")
    health = client.get("/health")
    assert health.status_code == 200 and health.json()["status"] == "degraded"
    response = client.post("/model/train")
    assert response.status_code == 422 and response.json()["code"] == "DATASET_ERROR"


def test_empty_list_before_first_training(client):
    response = client.post("/predict", json=[])
    assert response.status_code == 422 and response.json()["code"] == "EMPTY_INPUT"


def test_legacy_nonfinite_parameter_is_not_published(isolated_service):
    with TestClient(isolated_service) as client:
        assert client.post("/model/train").is_success
    artifact = joblib.load(main.MODEL_PATH)
    artifact["pipeline"].named_steps["classifier"].set_params(C=float("inf"))
    joblib.dump(artifact, main.MODEL_PATH)
    with TestClient(isolated_service) as client:
        status = client.get("/model/status").json()
        assert status["trained"] is False and status["load_error"]
