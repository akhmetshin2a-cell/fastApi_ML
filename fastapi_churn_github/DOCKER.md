# Запуск churn-сервиса в Docker

Нужен установленный и запущенный Docker Desktop. Все команды выполняются
из папки fastapi_churn_overall.

## Сборка и запуск

```bash
docker build -t fastapi-churn:day13 .
docker run -d --name fastapi-churn -p 127.0.0.1:8002:8000 -v churn-models:/app/models fastapi-churn:day13
```

Сервис доступен на компьютере по адресам:

- http://127.0.0.1:8002/health
- http://127.0.0.1:8002/docs

Внутри контейнера Uvicorn слушает порт 8000. На компьютере используется
8002, чтобы не мешать обычному учебному серверу на 8000 или 8001.

Образ содержит исходный CSV. Модель из macOS в образ не копируется:
первое обучение выполняется внутри Linux-контейнера. Обучите модель
через POST /model/train в /docs. Именованный том churn-models сохраняет
модель при удалении и повторном создании контейнера с тем же томом.
История обучений по-прежнему хранится в памяти; после перезапуска
восстанавливается только запись последней сохранённой модели.

## Проверка изнутри контейнера

```bash
docker exec fastapi-churn python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
docker exec fastapi-churn python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/docs').status)"
```

Первая команда должна вывести JSON, вторая — 200.
До обучения status=degraded, model_available=false, dataset_loaded=true.
После успешного обучения status=ok, model_available=true.

HTTP 200 у /health означает, что приложение отвечает. Docker HEALTHCHECK
проверяет доступность приложения: отсутствие модели до первого обучения
не делает контейнер неисправным. Состояние модели проверяется по JSON.

## Логи и управление

```bash
docker logs -f fastapi-churn
docker inspect --format '{{.State.Health.Status}}' fastapi-churn
docker restart fastapi-churn
docker stop fastapi-churn
```

В логах есть загрузка CSV, начало и завершение обучения, метрики,
число клиентов и время предсказания, а также ошибки. Полные признаки
клиентов и тела запросов в информационные логи не записываются.

Для локального запуска без Docker:

```bash
source .venv/bin/activate
python -m uvicorn main:app --reload --port 8001
```
