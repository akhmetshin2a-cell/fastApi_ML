"""Проверка вероятностей при старте сервиса и для каждого запроса."""
import joblib
import pytest
from fastapi.testclient import TestClient

import main


def test_saved_model_with_nan_is_rejected(isolated_service):
    with TestClient(isolated_service) as client:
        assert client.post("/model/train").status_code == 200
    artifact = joblib.load(main.MODEL_PATH)
    artifact["pipeline"].named_steps["classifier"].coef_[0, 0] = float("nan")
    joblib.dump(artifact, main.MODEL_PATH)

    with TestClient(isolated_service) as client:
        status = client.get("/model/status").json()
        assert status["trained"] is False and status["load_error"]
        assert client.get("/health").json()["status"] == "degraded"
        response = client.post("/predict", json=main.PREDICTION_EXAMPLE)
        assert response.status_code == 503 and response.json()["code"] == "MODEL_NOT_READY"


def test_predict_rejects_nan_in_active_model(client):
    assert client.post("/model/train").status_code == 200
    main.app.state.churn_model.named_steps["classifier"].coef_[0, 0] = float("nan")
    for payload in (main.PREDICTION_EXAMPLE, [main.PREDICTION_EXAMPLE] * 2):
        response = client.post("/predict", json=payload)
        assert response.status_code == 500 and response.json()["code"] == "PREDICTION_ERROR"


@pytest.mark.parametrize("matrix", [
    [[0.4, 0.6]],
    [[1], [1]],
    [[0.4, 0.6], [float("inf"), 0]],
    [[0.4, 0.6], [-0.1, 1.1]],
    [[0.4, 0.6], [0.3, 0.3]],
])
def test_invalid_probability_batch_is_rejected(client, monkeypatch, matrix):
    assert client.post("/model/train").status_code == 200
    monkeypatch.setattr(main.app.state.churn_model, "predict_proba", lambda frame: matrix)
    response = client.post("/predict", json=[main.PREDICTION_EXAMPLE] * 2)
    assert response.status_code == 500 and response.json()["code"] == "PREDICTION_ERROR"
