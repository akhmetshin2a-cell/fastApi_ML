"""Полный сценарий сервиса через TestClient, без сетевого сервера."""
import pytest
from fastapi.testclient import TestClient
import main


@pytest.mark.parametrize('config', [
    {'model_type': 'logreg', 'hyperparameters': {'C': 1.0}},
    {'model_type': 'random_forest', 'hyperparameters': {'n_estimators': 15, 'max_depth': 4}},
])
def test_full_workflow_and_reload(client, churn_rows, config):
    assert client.get('/model/status').json()['trained'] is False
    assert client.get('/model/metrics').json()['total'] == 0
    assert client.get('/dataset/info').json()['rows'] == 40
    assert client.get('/dataset/preview?n=2').json() == churn_rows[:2]
    response = client.post('/model/train', json=config)
    assert response.status_code == 200
    training = response.json()
    assert training['train_rows'] == 32
    assert training['test_rows'] == 8
    assert main.MODEL_PATH.exists()
    status = client.get('/model/status').json()
    assert status['trained'] is True
    assert status['model_type'] == config['model_type']
    for metric in ('accuracy', 'f1', 'roc_auc'):
        assert 0 <= training[metric] <= 1
        assert status['metrics'][metric] == training[metric]
    assert client.get('/model/metrics').json()['latest']['metrics'] == status['metrics']
    samples = [{k: v for k, v in row.items() if k != 'churn'} for row in churn_rows[:2]]
    prediction_response = client.post('/predict', json=samples)
    assert prediction_response.status_code == 200
    predictions = prediction_response.json()
    assert len(predictions) == 2
    for sample, expected, result in zip(samples, churn_rows[:2], predictions):
        assert result['churn'] == expected['churn']
        assert sum(result['probabilities'].values()) == pytest.approx(1)
        assert client.post('/predict', json=sample).json() == result
    # Новый lifespan восстанавливает модель из временного файла без обучения.
    with TestClient(main.app) as restarted:
        assert restarted.get('/model/status').json() == status
        assert restarted.post('/predict', json=samples).json() == predictions


def test_prediction_without_model(client):
    response = client.post('/predict', json=main.PREDICTION_EXAMPLE)
    assert response.status_code == 503
    assert response.json()['code'] == 'MODEL_NOT_READY'
    assert set(response.json()) == {'code', 'message', 'details'}


@pytest.mark.parametrize('payload', [
    {}, {**main.PREDICTION_EXAMPLE, 'usage_hours': 'wrong'},
    {**main.PREDICTION_EXAMPLE, 'extra': 5},
])
def test_invalid_features(client, payload):
    response = client.post('/predict', json=payload)
    assert response.status_code == 422
    assert response.json()['code'] == 'VALIDATION_ERROR'


@pytest.mark.parametrize('mode,status,code', [
    ('missing', 404, 'DATASET_NOT_FOUND'),
    ('empty', 422, 'EMPTY_DATASET'),
    ('invalid', 422, 'DATASET_ERROR'),
])
def test_training_dataset_errors(client, csv_path, mode, status, code):
    if mode == 'missing':
        csv_path.unlink()
    elif mode == 'empty':
        csv_path.write_text(','.join(main.DatasetRowChurn.model_fields) + '\n')
    else:
        csv_path.write_text('wrong,churn\n1,0\n')
    response = client.post('/model/train', json={'model_type': 'logreg'})
    assert response.status_code == status
    assert response.json()['code'] == code
    assert client.get('/model/status').json()['trained'] is False
    assert client.get('/model/metrics').json()['total'] == 0


def test_failed_retraining_preserves_saved_model(client):
    assert client.post('/model/train').status_code == 200
    before = client.get('/model/status').json()
    saved = main.MODEL_PATH.read_bytes()
    response = client.post('/model/train', json={'hyperparameters': {'C': -1}})
    assert response.status_code == 422
    assert response.json()['code'] == 'TRAINING_CONFIG_ERROR'
    assert client.get('/model/status').json() == before
    assert main.MODEL_PATH.read_bytes() == saved
