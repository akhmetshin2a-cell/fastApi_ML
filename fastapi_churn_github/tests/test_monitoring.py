"""Диагностика и журнал ключевых событий сервиса."""
import logging
import pytest
import main


def test_health_before_and_after_training(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {
        'status': 'degraded', 'model_available': False,
        'dataset_loaded': True, 'dataset_rows': 40, 'dataset_error': None,
    }
    assert client.post('/model/train').status_code == 200
    health = client.get('/health').json()
    assert health['status'] == 'ok'
    assert health['model_available'] is True
    assert health['dataset_loaded'] is True
    assert client.get('/docs').status_code == 200
    assert '/health' in client.get('/openapi.json').json()['paths']


@pytest.mark.parametrize('mode', ['missing', 'empty', 'invalid'])
def test_health_handles_unusable_dataset(client, csv_path, mode):
    if mode == 'missing':
        csv_path.unlink()
    elif mode == 'empty':
        csv_path.write_text(','.join(main.DatasetRowChurn.model_fields) + '\n')
    else:
        csv_path.write_text('wrong,churn\n1,0\n')
    response = client.get('/health')
    assert response.status_code == 200
    health = response.json()
    assert health['status'] == 'degraded'
    assert health['dataset_loaded'] is False
    assert health['dataset_rows'] == 0
    assert health['dataset_error']


def test_model_can_predict_when_dataset_removed(client, csv_path):
    assert client.post('/model/train').status_code == 200
    csv_path.unlink()
    health = client.get('/health').json()
    assert health['model_available'] is True
    assert health['dataset_loaded'] is False
    assert health['status'] == 'degraded'
    assert client.post('/predict', json=main.PREDICTION_EXAMPLE).status_code == 200


def test_key_events_logged_without_client_features(client, caplog):
    caplog.set_level(logging.INFO, logger='churn_service')
    assert client.get('/dataset/info').status_code == 200
    assert client.post('/model/train').status_code == 200
    assert client.post('/predict', json=main.PREDICTION_EXAMPLE).status_code == 200
    assert client.post('/predict', json={}).status_code == 422
    records = [record.getMessage() for record in caplog.records if record.name == 'churn_service']
    for event in ('dataset_load_started', 'dataset_load_completed', 'training_started',
                  'training_completed', 'predict_started', 'predict_completed', 'request_validation_failed'):
        assert any(event in message for message in records)
    assert not any('monthly_fee' in message or 'usage_hours' in message for message in records)
