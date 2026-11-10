"""Контрольные случаи на временных CSV и артефактах модели."""
import csv

import joblib
import pytest
from fastapi.testclient import TestClient

import main


def test_forest_handles_nearly_constant_numeric_feature(isolated_service, csv_path, churn_rows):
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=churn_rows[0])
        writer.writeheader()
        for index, row in enumerate(churn_rows):
            writer.writerow({**row, "usage_hours": (index % 2) * 1e-45})

    sample = {**main.PREDICTION_EXAMPLE, "usage_hours": 40}
    with TestClient(isolated_service) as client:
        assert client.post("/model/train", json={"model_type": "random_forest"}).status_code == 200
        prediction = client.post("/predict", json=sample)
        assert prediction.status_code == 200
        assert sum(prediction.json()["probabilities"].values()) == pytest.approx(1)
    with TestClient(isolated_service) as restarted:
        assert restarted.post("/predict", json=sample).json() == prediction.json()


@pytest.mark.parametrize("kind", ["logreg", "random_forest"])
def test_incompatible_classifier_width_on_startup(isolated_service, kind):
    with TestClient(isolated_service) as client:
        assert client.post("/model/train", json={"model_type": kind}).status_code == 200
    artifact = joblib.load(main.MODEL_PATH)
    artifact["pipeline"].named_steps["classifier"].n_features_in_ += 1
    joblib.dump(artifact, main.MODEL_PATH)
    with TestClient(isolated_service) as restarted:
        status = restarted.get("/model/status").json()
        assert status["trained"] is False and status["load_error"]
        assert restarted.get("/health").json()["status"] == "degraded"
        assert restarted.post("/predict", json=main.PREDICTION_EXAMPLE).status_code == 503


def test_overflowing_integer_parameter_preserves_previous_model(client):
    assert client.post("/model/train").status_code == 200
    before = main.MODEL_PATH.read_bytes()
    history = client.get("/model/metrics").json()
    status = client.get("/model/status").json()
    result = client.post("/model/train", json={"hyperparameters": {"C": 10**320}})
    assert result.status_code == 422 and result.json()["code"] == "TRAINING_CONFIG_ERROR"
    assert main.MODEL_PATH.read_bytes() == before
    assert client.get("/model/status").json() == status
    assert client.get("/model/metrics").json() == history


@pytest.mark.parametrize("field", ["list[FeatureVectorChurn]", "FeatureVectorChurn"])
def test_training_reports_extra_field_named_like_prediction_branch(client, field):
    result = client.post("/model/train", json={field: 123})
    assert result.status_code == 422
    details = result.json()["details"]
    assert [(item["location"], item["type"]) for item in details] == [(["body", field], "extra_forbidden")]
