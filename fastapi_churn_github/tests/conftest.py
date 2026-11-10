"""Повторяемые тестовые данные и изолированный TestClient."""
import csv

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import State
import main


@pytest.fixture
def churn_rows():
    """40 фиксированных клиентов, по 20 на класс, без случайной генерации."""
    rows = []
    for i in range(40):
        churn = i % 2
        rows.append({
            'monthly_fee': float(20 + i % 7 + 40 * churn),
            'usage_hours': float(80 - 60 * churn + i % 5),
            'support_requests': 1 + 5 * churn,
            'account_age_months': 36 - 30 * churn + i % 3,
            'failed_payments': 2 * churn,
            'region': ['europe', 'asia', 'america', 'africa'][i % 4],
            'device_type': ['mobile', 'desktop', 'tablet'][i % 3],
            'payment_method': ['card', 'paypal', 'crypto'][i % 3],
            'autopay_enabled': 1 - churn,
            'churn': churn,
        })
    return rows


@pytest.fixture
def csv_path(tmp_path, churn_rows):
    path = tmp_path / 'data' / 'churn_dataset.csv'
    path.parent.mkdir()
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(churn_rows[0]))
        writer.writeheader()
        writer.writerows(churn_rows)
    return path


@pytest.fixture
def isolated_service(monkeypatch, tmp_path, csv_path):
    """Файл модели и состояние приложения принадлежат только этому тесту."""
    monkeypatch.setattr(main, 'DATASET_PATH', csv_path)
    monkeypatch.setattr(main, 'MODEL_PATH', tmp_path / 'models' / 'churn_model.joblib')
    # Заменяем весь State: TestClient управляет его жизненным циклом.
    monkeypatch.setattr(main.app, "state", State(dict(
        churn_model=None, model_artifact=None, model_load_error=None, training_history=[],
    )))
    return main.app


@pytest.fixture
def client(isolated_service):
    # Контекст запускает lifespan: проверяем настоящее поведение старта сервиса.
    with TestClient(isolated_service) as test_client:
        yield test_client
