FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

RUN useradd --create-home --uid 10001 churn \
    && mkdir -p /app/models \
    && chown churn:churn /app/models
COPY --chown=churn:churn main.py ./
COPY --chown=churn:churn data/churn_dataset.csv ./data/churn_dataset.csv
USER churn

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).close()"

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
