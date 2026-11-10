"""Учебный API предсказания оттока клиентов.

Весь код сервиса находится в main.py по выбранной структуре проекта.
Блоки: конфигурация, схемы, данные, ML pipeline, хранение, ошибки и API.
Запуск: python -m uvicorn main:app --reload
"""

import csv
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Annotated, Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field, ValidationError, field_validator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


# ----- Конфигурация -----
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "data" / "churn_dataset.csv"
MODEL_PATH = PROJECT_DIR / "models" / "churn_model.joblib"
logger = logging.getLogger("churn_service")
logger.setLevel(logging.INFO)

TARGET_COLUMN = "churn"
NUMERIC_FEATURES = [
    "monthly_fee", "usage_hours", "support_requests",
    "account_age_months", "failed_payments", "autopay_enabled",
]
CATEGORICAL_FEATURES = ["region", "device_type", "payment_method"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ----- Схемы данных -----
# Пределы оставляют запас для масштабирования и преобразования лесом в float32.
FeatureFloat = Annotated[float, Field(ge=-1e12, le=1e12)]
FeatureCount = Annotated[int, Field(ge=-(2**53 - 1), le=2**53 - 1)]
AutopayFlag = Annotated[int, Field(ge=0, le=1)]


class FeatureVectorChurn(BaseModel):
    """Признаки одного клиента для предсказания оттока."""

    model_config = {"allow_inf_nan": False, "extra": "forbid"}

    monthly_fee: FeatureFloat
    usage_hours: FeatureFloat
    support_requests: FeatureCount
    account_age_months: FeatureCount
    failed_payments: FeatureCount
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: AutopayFlag


class DatasetRowChurn(FeatureVectorChurn):
    """Строка обучающего датасета: признаки и известный ответ churn."""

    churn: int


class DatasetRowWithMissingChurn(DatasetRowChurn):
    """Для CSV допускаем пропуски признаков; целевой churn обязателен."""

    model_config = {"allow_inf_nan": False}

    monthly_fee: FeatureFloat | None = None
    usage_hours: FeatureFloat | None = None
    support_requests: FeatureCount | None = None
    account_age_months: FeatureCount | None = None
    failed_payments: FeatureCount | None = None
    region: str | None = None
    device_type: str | None = None
    payment_method: str | None = None
    autopay_enabled: AutopayFlag | None = None


# ----- Настройки обучения -----
class TrainingConfigChurn(BaseModel):
    """Выбор алгоритма и его настроек до начала обучения."""

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {"examples": [
            {"model_type": "logreg", "hyperparameters": {"C": 1.0, "max_iter": 1000}},
            {"model_type": "random_forest", "hyperparameters": {"n_estimators": 100, "max_depth": 5, "random_state": 42}},
        ]},
    }

    model_type: Literal["logreg", "random_forest"] = "logreg"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("hyperparameters")
    @classmethod
    def validate_json_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        # JSON-сериализация с allow_nan=False проверяет также вложенные списки и словари.
        json.dumps(value, allow_nan=False)
        return value


def create_churn_classifier(config: TrainingConfigChurn):
    """Создать выбранный классификатор; неизвестные параметры — ошибка."""
    if config.model_type == "logreg":
        classifier = LogisticRegression(max_iter=1000, random_state=42)
    else:
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    unknown = set(config.hyperparameters) - set(classifier.get_params(deep=False))
    if unknown:
        raise ValueError(f"Неизвестные гиперпараметры {config.model_type}: {', '.join(sorted(unknown))}")
    parameters = config.hyperparameters.copy()
    weights = parameters.get("class_weight")
    if isinstance(weights, dict):
        if not all(str(key) in {"0", "1"} for key in weights):
            raise ValueError("В class_weight допустимы только классы 0 и 1.")
        parameters["class_weight"] = {int(key): weight for key, weight in weights.items()}
    classifier.set_params(**parameters)
    return classifier


def model_metadata(pipeline: Pipeline) -> dict:
    """Фактический тип и параметры классификатора, включая значения по умолчанию."""
    classifier = pipeline.named_steps["classifier"]
    if isinstance(classifier, LogisticRegression):
        model_type = "logreg"
    elif isinstance(classifier, RandomForestClassifier):
        model_type = "random_forest"
    else:
        raise ValueError("Неизвестный тип сохранённого классификатора.")
    return {"model_type": model_type, "hyperparameters": classifier.get_params(deep=False)}


# ----- Схема ответа и примеры для документации -----
class PredictionResponseChurn(BaseModel):
    """Предсказанный класс и оценки вероятностей обоих классов."""

    churn: Literal[0, 1] = Field(description="0 — останется, 1 — уйдёт")
    probabilities: dict[str, float] = Field(
        description='Вероятности: ключ "0" — останется, ключ "1" — уйдёт. Сумма равна 1.'
    )


PREDICTION_EXAMPLE = {
    "monthly_fee": 29.9,
    "usage_hours": 40.5,
    "support_requests": 2,
    "account_age_months": 12,
    "failed_payments": 0,
    "region": "europe",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 1,
}
SECOND_PREDICTION_EXAMPLE = {
    "monthly_fee": 99.9,
    "usage_hours": 1.0,
    "support_requests": 10,
    "account_age_months": 1,
    "failed_payments": 4,
    "region": "asia",
    "device_type": "desktop",
    "payment_method": "crypto",
    "autopay_enabled": 0,
}


# ----- Чтение CSV -----


class DatasetError(ValueError):
    """CSV имеет неподходящую структуру или некорректные значения."""


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    """Прочитать CSV в список словарей с проверенными типами значений."""
    logger.info("dataset_load_started path=%s", path)
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        expected = set(DatasetRowChurn.model_fields)
        try:
            headers = reader.fieldnames
            if headers is None or len(headers) != len(expected) or set(headers) != expected:
                raise DatasetError("CSV должен содержать девять признаков и churn без повторяющихся столбцов.")
            for line_number, raw_row in enumerate(reader, start=2):
                if None in raw_row or any(value is None for value in raw_row.values()):
                    raise DatasetError(f"Неверное количество значений в строке {line_number}.")
                # CSV содержит строки; Pydantic преобразует числа в float и int.
                # Пустые ячейки и распространённые маркеры пропусков -> None.
                normalized_row = {
                    key: None if value.strip().lower() in {"", "nan", "na", "null", "none"} else value
                    for key, value in raw_row.items()
                }
                validated_row = DatasetRowWithMissingChurn.model_validate(normalized_row)
                if validated_row.churn not in (0, 1):
                    raise DatasetError(f"churn должен быть 0 или 1 в строке {line_number}.")
                rows.append(validated_row.model_dump())
        except (ValidationError, csv.Error) as exc:
            raise DatasetError("Ошибка данных CSV.") from exc
    logger.info("dataset_load_completed rows=%s columns=%s", len(rows), len(expected))
    return rows


# ----- Единая подготовка таблицы признаков -----
def build_feature_frame(rows: list[dict]) -> pd.DataFrame:
    """Одинаковый набор, порядок и числовые типы при обучении и предсказании."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise DatasetError("Список клиентов пуст.")
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    extra = set(frame.columns) - set(FEATURE_COLUMNS) - {TARGET_COLUMN}
    if missing or extra:
        raise DatasetError(f"Неверный набор признаков. Отсутствуют: {sorted(missing)}; лишние: {sorted(extra)}.")
    # Порядок ключей исходного JSON не важен: столбцы выбираются по именам.
    X = frame[FEATURE_COLUMNS].copy().fillna(float("nan"))
    for name in NUMERIC_FEATURES:
        X[name] = pd.to_numeric(X[name], errors="raise")
    return X


def build_churn_preprocessor(*, scale_numeric: bool = True) -> ColumnTransformer:
    """Создать необученную предобработку; fit будет только на train."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ], remainder="drop")


@dataclass
class PreparedData:
    """Подготовленные выборки и обработчик, обученный только на train."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    imputer: ColumnTransformer | None


def prepare_data(rows: list[dict], test_size: float = 0.2,
                 random_state: int = 42, impute: bool = True) -> PreparedData:
    """Разбить строки на train/test и заполнить пропуски без утечки из test."""
    if not 0 < test_size < 1:
        raise DatasetError("test_size должен быть между 0 и 1.")
    frame = pd.DataFrame(rows)
    if frame.empty or not set(FEATURE_COLUMNS + [TARGET_COLUMN]).issubset(frame.columns):
        raise DatasetError("Для разбиения нужны непустые данные со всеми столбцами.")

    # X — только входные признаки; y — известные ответы.
    X = build_feature_frame(rows)
    y = frame[TARGET_COLUMN].copy()
    if y.isna().any() or not y.isin([0, 1]).all() or y.nunique() != 2:
        raise DatasetError("Нужны оба класса churn (0 и 1), без пропусков цели.")
    y = y.astype(int)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y,
        )
    except ValueError as exc:
        raise DatasetError("Недостаточно данных каждого класса для разбиения train/test.") from exc
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise DatasetError("В train и test нужны оба класса churn. Добавьте данные или измените размер test.")

    # Полностью пустой столбец не имеет медианы или моды: сообщаем об этом явно.
    empty_columns = X_train.columns[X_train.isna().all()].tolist()
    if empty_columns:
        raise DatasetError(f"В train полностью пустые признаки: {empty_columns}.")

    # При обучении пропуски обрабатывает единый pipeline.
    if not impute:
        return PreparedData(X_train, X_test, y_train, y_test, None)

    imputer = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("categorical", SimpleImputer(strategy="most_frequent"), CATEGORICAL_FEATURES),
    ])
    # Статистики вычисляются только на train. На test применяем уже готовые.
    train_values = imputer.fit_transform(X_train)
    test_values = imputer.transform(X_test)
    train_frame = pd.DataFrame(train_values, columns=FEATURE_COLUMNS, index=X_train.index)
    test_frame = pd.DataFrame(test_values, columns=FEATURE_COLUMNS, index=X_test.index)
    for name in NUMERIC_FEATURES:
        train_frame[name] = pd.to_numeric(train_frame[name])
        test_frame[name] = pd.to_numeric(test_frame[name])
    return PreparedData(
        X_train=train_frame,
        X_test=test_frame,
        y_train=y_train,
        y_test=y_test,
        imputer=imputer,
    )


def class_distribution(y: pd.Series) -> dict:
    """Вернуть количество и долю объектов каждого класса."""
    return {
        str(label): {"count": int((y == label).sum()), "ratio": float((y == label).mean())}
        for label in (0, 1)
    }


# ----- Обучение полного pipeline -----
def train_churn_model(training_rows: list[dict],
                      config: TrainingConfigChurn | None = None) -> Pipeline:
    """Обучить pipeline только на тренировочных строках, без тестовых данных."""
    frame = pd.DataFrame(training_rows)
    if frame.empty or not set(FEATURE_COLUMNS + [TARGET_COLUMN]).issubset(frame.columns):
        raise DatasetError("Тренировочный датасет пуст или не содержит нужных столбцов.")
    X = build_feature_frame(training_rows)
    y = frame[TARGET_COLUMN]
    if y.isna().any() or not y.isin([0, 1]).all() or y.nunique() != 2:
        raise DatasetError("Для обучения нужны оба класса churn: 0 и 1, без пропусков.")
    empty_columns = X.columns[X.isna().all()].tolist()
    if empty_columns:
        raise DatasetError(f"В train полностью пустые признаки: {empty_columns}.")

    classifier = create_churn_classifier(config or TrainingConfigChurn())
    preprocessor = build_churn_preprocessor(scale_numeric=isinstance(classifier, LogisticRegression))
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    pipeline.fit(X, y.astype(int))
    return pipeline


# ----- Сохранение и загрузка модели -----


def save_churn_model(model: Pipeline, trained_at: str, metrics: dict,
                     path: Path | None = None) -> None:
    """Сохранить весь pipeline, дату обучения и метрики одним файлом."""
    path = MODEL_PATH if path is None else path
    artifact = {"pipeline": model, "trained_at": trained_at, "metrics": metrics, **model_metadata(model)}
    path.parent.mkdir(parents=True, exist_ok=True)
    # Сначала пишем временный файл: при сбое предыдущая модель останется целой.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".joblib", delete=False) as file:
            temporary_path = Path(file.name)
        joblib.dump(artifact, temporary_path, compress=3)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def checked_probabilities(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Получить по две вероятности на клиента и проверить каждую строку."""
    values = np.asarray(model.predict_proba(frame))
    if values.shape != (len(frame), 2) or values.dtype.kind not in "iuf":
        raise ValueError("Ожидаются две числовые вероятности на каждого клиента.")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Вероятности должны быть конечными числами от 0 до 1.")
    if not np.allclose(values.sum(axis=1, dtype=np.float64), 1.0, rtol=0, atol=1e-6):
        raise ValueError("Сумма вероятностей каждого клиента должна равняться 1.")
    return values


def load_churn_model(path: Path | None = None) -> dict | None:
    """Вернуть модель с метаданными; отсутствие файла нормально до обучения."""
    path = MODEL_PATH if path is None else path
    if not path.exists():
        return None
    # Загружаем только наш локальный файл, созданный save_churn_model.
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("pipeline"), Pipeline):
        raise ValueError("В файле отсутствует pipeline модели.")
    pipeline = artifact["pipeline"]
    if not {"preprocessor", "classifier"}.issubset(pipeline.named_steps):
        raise ValueError("Неверная структура pipeline.")
    check_is_fitted(pipeline.named_steps["classifier"])
    check_is_fitted(pipeline.named_steps["preprocessor"])
    classes = pipeline.classes_
    integer_labels = all(isinstance(label, Integral) and not isinstance(label, bool) for label in classes)
    if not integer_labels or len(classes) != 2 or set(classes) != {0, 1}:
        raise ValueError("Модель должна предсказывать целочисленные классы 0 и 1.")
    if list(pipeline.feature_names_in_) != FEATURE_COLUMNS:
        raise ValueError("Схема сохранённой модели отличается от текущей. Обучите модель заново.")
    # Проверяем совместимость этапов на допустимых границах входа. Это также
    # выявляет старые леса, у которых scaler переполняет float32 при предсказании.
    properties = FeatureVectorChurn.model_json_schema()["properties"]
    boundary_rows = [
        {**PREDICTION_EXAMPLE, **{name: properties[name][bound] for name in NUMERIC_FEATURES}}
        for bound in ("minimum", "maximum")
    ]
    checked_probabilities(pipeline, build_feature_frame(boundary_rows))
    trained_at = datetime.fromisoformat(artifact["trained_at"])
    if trained_at.tzinfo is None:
        raise ValueError("Дата обучения должна содержать часовой пояс.")
    metrics = artifact["metrics"]
    if not isinstance(metrics, dict) or not {"accuracy", "f1"}.issubset(metrics):
        raise ValueError("В файле отсутствуют обязательные метрики модели.")
    for value in metrics.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError("Некорректные метрики модели.")
    metrics.setdefault("roc_auc", None)  # В файлах прежнего формата AUC не измерен.
    # Поддерживаем старый формат файла: настройки читаем из классификатора.
    metadata = model_metadata(pipeline)
    TrainingConfigChurn.model_validate(metadata)
    artifact.update(metadata)
    return artifact


# ----- История обучений в памяти текущего процесса -----
training_lock = RLock()


def training_record(artifact: dict) -> dict:
    """Снимок результатов: последующие изменения не изменяют старую запись."""
    return deepcopy({
        "timestamp": artifact["trained_at"],
        "model_type": artifact["model_type"],
        "hyperparameters": artifact["hyperparameters"],
        "metrics": artifact["metrics"],
    })


@asynccontextmanager
async def lifespan(application: FastAPI):
    """При старте восстановить модель без повторного обучения."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger.info("service_starting")
    application.state.churn_model = None
    application.state.model_artifact = None
    application.state.model_load_error = None
    application.state.training_history = []
    try:
        artifact = load_churn_model()
        if artifact is not None:
            application.state.churn_model = artifact["pipeline"]
            application.state.model_artifact = artifact
            # Из файла восстанавливаем последнее обучение; остальные записи живут в памяти.
            application.state.training_history = [training_record(artifact)]
            logger.info("model_loaded model_type=%s trained_at=%s", artifact["model_type"], artifact["trained_at"])
        else:
            logger.info("model_not_found training_required=true")
    except Exception:
        # Повреждённый файл не мешает открыть API и заново обучить модель.
        logger.exception("Не удалось загрузить сохранённую churn модель")
        application.state.model_load_error = "Не удалось загрузить файл модели. Запустите POST /model/train."
    yield
    logger.info("service_stopped")


# ----- Единый формат ошибок -----
class ErrorResponseChurn(BaseModel):
    code: str
    message: str
    details: Any = None


class ServiceError(HTTPException):
    """Контролируемая ошибка с понятным кодом для клиента API."""

    def __init__(self, status_code: int, code: str, message: str, details: Any = None):
        super().__init__(status_code=status_code, detail=message)
        self.code = code
        self.details = details


class PredictionError(Exception):
    """Внутренняя ошибка выполнения модели; подробности остаются в журнале."""


def error_documentation(description: str, code: str, message: str,
                        details: Any = None) -> dict:
    return {
        "model": ErrorResponseChurn,
        "description": description,
        "content": {"application/json": {"examples": {
            "error": {"value": {"code": code, "message": message, "details": details}},
        }}},
    }


VALIDATION_ERROR_DOC = error_documentation(
    "Неверные поля, типы или значения запроса", "VALIDATION_ERROR",
    "Проверьте поля и типы данных запроса.",
    [{"location": ["body", "monthly_fee"], "type": "float_parsing", "message": "Ожидается число."}],
)
INTERNAL_ERROR_DOC = error_documentation(
    "Внутренняя ошибка сервиса", "INTERNAL_ERROR", "Не удалось выполнить запрос. Повторите попытку позже.",
)


# ----- Маршруты API -----
app = FastAPI(lifespan=lifespan, responses={422: VALIDATION_ERROR_DOC, 500: INTERNAL_ERROR_DOC})
app.state.churn_model = None
app.state.model_artifact = None
app.state.model_load_error = None
app.state.training_history = []


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Единый ответ и для наших ошибок, и для стандартных 404/405."""
    default_codes = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED", 422: "INVALID_DATA", 503: "MODEL_NOT_READY"}
    code = getattr(exc, "code", default_codes.get(exc.status_code, "HTTP_ERROR"))
    message = exc.detail if isinstance(exc.detail, str) else "Не удалось выполнить запрос."
    logger.warning("http_error path=%s status=%s code=%s", request.url.path, exc.status_code, code)
    return JSONResponse(status_code=exc.status_code, headers=exc.headers, content={
        "code": code, "message": message, "details": getattr(exc, "details", None),
    })


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, exc: RequestValidationError):
    # Возвращаем адрес и причину ошибки, без исходных данных и внутренних объектов.
    logger.warning("request_validation_failed path=%s errors=%s", request.url.path, len(exc.errors()))
    details = []
    for error in exc.errors():
        location = list(error["loc"])
        if request.scope.get("endpoint") is predict and len(location) >= 2 and location[0] == "body":
            branch = location[1]
            if branch in ("FeatureVectorChurn", "list[FeatureVectorChurn]"):
                if not isinstance(exc.body, (dict, list)):
                    details = [{"location": ["body"], "type": "invalid_input",
                                "message": "Ожидается объект клиента или список клиентов."}]
                    break
                if (branch == "FeatureVectorChurn") != isinstance(exc.body, dict):
                    continue
                del location[1]
        details.append({"location": location, "type": error["type"], "message": error["msg"]})
    return JSONResponse(status_code=422, content={
        "code": "VALIDATION_ERROR", "message": "Проверьте поля и типы данных запроса.", "details": details,
    })


@app.exception_handler(DatasetError)
async def dataset_error_handler(request: Request, exc: DatasetError):
    logger.warning("dataset_error path=%s reason=%s", request.url.path, exc)
    return JSONResponse(status_code=422, content={
        "code": "DATASET_ERROR", "message": str(exc), "details": None,
    })


@app.exception_handler(PredictionError)
async def prediction_error_handler(request: Request, exc: PredictionError):
    return JSONResponse(status_code=500, content={
        "code": "PREDICTION_ERROR", "message": "Не удалось выполнить предсказание. Проверьте статус модели или обучите её заново.", "details": None,
    })


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.error("Необработанная ошибка запроса %s", request.url.path, exc_info=(type(exc), exc, exc.__traceback__))
    return JSONResponse(status_code=500, content={
        "code": "INTERNAL_ERROR", "message": "Не удалось выполнить запрос. Повторите попытку позже.", "details": None,
    })


@app.get("/")
def read_root() -> dict[str, str]:
    """Подтвердить, что сервис работает. FastAPI превратит словарь в JSON."""
    return {"message": "ml churn service is running"}


@app.post(
    "/predict",
    response_model=PredictionResponseChurn | list[PredictionResponseChurn],
    summary="Предсказать отток для одного или нескольких клиентов",
    responses={
        200: {
            "description": "Один объект для одного клиента; список в порядке входных клиентов для пакета. Числа в примерах условные.",
            "content": {"application/json": {"examples": {
                "single": {"summary": "Один результат", "value": {"churn": 0, "probabilities": {"0": 0.8, "1": 0.2}}},
                "batch": {"summary": "Список результатов", "value": [
                    {"churn": 0, "probabilities": {"0": 0.8, "1": 0.2}},
                    {"churn": 1, "probabilities": {"0": 0.3, "1": 0.7}},
                ]},
            }}},
        },
        503: error_documentation("Модель недоступна", "MODEL_NOT_READY", "Модель не обучена или не загружена. Сначала выполните POST /model/train."),
        422: VALIDATION_ERROR_DOC,
        500: error_documentation("Сбой предсказания", "PREDICTION_ERROR", "Не удалось выполнить предсказание. Проверьте статус модели или обучите её заново."),
    },
)
def predict(
    features: FeatureVectorChurn | list[FeatureVectorChurn] = Body(
        openapi_examples={
            "single": {"summary": "Один клиент", "value": PREDICTION_EXAMPLE},
            "batch": {"summary": "Два клиента", "value": [PREDICTION_EXAMPLE, SECOND_PREDICTION_EXAMPLE]},
        }
    ),
) -> PredictionResponseChurn | list[PredictionResponseChurn]:
    """Применить сохранённый pipeline без повторного обучения.

    Передайте JSON-объект или непустой массив объектов с девятью признаками.
    churn передавать не нужно: его предсказывает модель.
    Вероятности — оценки модели, а не гарантия будущего поведения клиента.
    """
    single = isinstance(features, FeatureVectorChurn)
    clients = [features] if single else features
    if not clients:
        raise ServiceError(422, "EMPTY_INPUT", "Список клиентов не должен быть пустым.")
    # Запоминаем ссылку: все клиенты запроса обрабатываются одной версией модели.
    model = app.state.churn_model
    if model is None:
        raise ServiceError(503, "MODEL_NOT_READY", "Модель не обучена или не загружена. Сначала выполните POST /model/train.")
    # model_dump превращает Pydantic-объекты в строки таблицы для pipeline.
    started = perf_counter()
    logger.info("predict_started clients=%s", len(clients))
    frame = build_feature_frame([client.model_dump() for client in clients])
    try:
        labels = model.predict(frame)
        probabilities = checked_probabilities(model, frame)
        # Порядок столбцов predict_proba соответствует model.classes_.
        classes = model.classes_
        results = [
            PredictionResponseChurn(
                churn=int(label),
                probabilities={str(int(cls)): float(value) for cls, value in zip(classes, values)},
            )
            for label, values in zip(labels, probabilities)
        ]
        logger.info("predict_completed clients=%s duration_ms=%.2f", len(clients), (perf_counter() - started) * 1000)
        return results[0] if single else results
    except Exception as exc:
        logger.exception("Ошибка предсказания churn")
        raise PredictionError() from exc


def get_dataset() -> list[dict]:
    """Загрузить данные или вернуть понятную ошибку API."""
    try:
        return load_dataset(DATASET_PATH)
    except FileNotFoundError as exc:
        raise ServiceError(404, "DATASET_NOT_FOUND", "Файл churn_dataset.csv не найден в папке data.") from exc
    except DatasetError as exc:
        raise ServiceError(422, "DATASET_ERROR", str(exc)) from exc
    except (OSError, UnicodeError) as exc:
        raise ServiceError(500, "DATASET_READ_ERROR", "Не удалось прочитать датасет. Проверьте CSV и типы данных.") from exc


@app.get("/dataset/preview", response_model=list[DatasetRowWithMissingChurn])
def dataset_preview(n: int = Query(default=5, ge=1, le=100)) -> list[dict]:
    """Вернуть первые n строк. Допустимо от 1 до 100 строк."""
    rows = get_dataset()
    return rows[:n]


@app.get("/dataset/info")
def dataset_info() -> dict:
    """Показать размер, названия столбцов и число клиентов каждого класса."""
    rows = get_dataset()
    columns = list(DatasetRowChurn.model_fields)
    counts = {"0": 0, "1": 0}
    for row in rows:
        counts[str(row[TARGET_COLUMN])] += 1
    return {
        "rows": len(rows),
        "columns": len(columns),
        "column_names": columns,
        "feature_names": [name for name in columns if name != TARGET_COLUMN],
        "churn_distribution": counts,
    }


@app.get("/dataset/split-info")
def dataset_split_info() -> dict:
    """Проверить подготовку данных: 80% train, 20% test, random_state=42."""
    rows = get_dataset()
    try:
        prepared = prepare_data(rows)
    except DatasetError as exc:
        raise ServiceError(422, "DATASET_ERROR", str(exc)) from exc
    return {
        "test_size": 0.2,
        "random_state": 42,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "train": {
            "rows": len(prepared.y_train),
            "features": prepared.X_train.shape[1],
            "churn_distribution": class_distribution(prepared.y_train),
        },
        "test": {
            "rows": len(prepared.y_test),
            "features": prepared.X_test.shape[1],
            "churn_distribution": class_distribution(prepared.y_test),
        },
    }


@app.post("/model/train", responses={
    404: error_documentation("CSV отсутствует", "DATASET_NOT_FOUND", "Файл churn_dataset.csv не найден в папке data."),
    422: error_documentation("Пустой датасет или неверные настройки обучения", "EMPTY_DATASET", "Датасет пуст. Добавьте строки в data/churn_dataset.csv."),
    500: error_documentation("Сбой сохранения модели", "MODEL_SAVE_ERROR", "Не удалось сохранить модель на диск."),
})
def model_train(config: TrainingConfigChurn | None = None) -> dict:
    """Обучить выбранную модель на 80% строк и оценить на оставшихся 20%.

    Без тела запроса используется logreg с настройками по умолчанию.
    Неизвестный тип модели или неверные гиперпараметры возвращают 422.
    """
    started = perf_counter()
    logger.info("training_started model_type=%s", (config or TrainingConfigChurn()).model_type)
    rows = get_dataset()
    if not rows:
        raise ServiceError(422, "EMPTY_DATASET", "Датасет пуст. Добавьте строки в data/churn_dataset.csv.")
    try:
        # Сначала разбиение сырых данных; вся предобработка обучается на train.
        split = prepare_data(rows, impute=False)
        training_rows = split.X_train.assign(churn=split.y_train).to_dict(orient="records")
        pipeline = train_churn_model(training_rows, config)
        predictions = pipeline.predict(split.X_test)
        positive_index = list(pipeline.classes_).index(1)
        probabilities = pipeline.predict_proba(split.X_test)[:, positive_index]
        metrics = {
            "accuracy": float(accuracy_score(split.y_test, predictions)),
            "f1": float(f1_score(split.y_test, predictions, pos_label=1, zero_division=0)),
            "roc_auc": float(roc_auc_score(split.y_test, probabilities)),
        }
    except DatasetError:
        raise
    except (ValueError, TypeError, OverflowError) as exc:
        logger.warning("Неверная конфигурация обучения: %s", exc)
        raise ServiceError(422, "TRAINING_CONFIG_ERROR", "Неверные настройки обучения. Проверьте названия, типы и значения гиперпараметров.") from exc
    # Сохранение, смена активной модели и запись истории — одна секция.
    # При ошибке обучения или записи предыдущая модель и история сохраняются.
    with training_lock:
        trained_at = datetime.now(timezone.utc).isoformat()
        try:
            save_churn_model(pipeline, trained_at, metrics)
        except Exception as exc:
            logger.exception("Не удалось сохранить churn модель")
            raise ServiceError(500, "MODEL_SAVE_ERROR", "Не удалось сохранить модель на диск.") from exc
        artifact = {"pipeline": pipeline, "trained_at": trained_at, "metrics": metrics, **model_metadata(pipeline)}
        app.state.model_artifact = artifact
        app.state.churn_model = pipeline
        app.state.model_load_error = None
        app.state.training_history.append(training_record(artifact))
    logger.info("training_completed model_type=%s train_rows=%s test_rows=%s metrics=%s duration_ms=%.2f",
                artifact["model_type"], len(split.y_train), len(split.y_test), metrics,
                (perf_counter() - started) * 1000)
    return {
        "model": type(pipeline.named_steps["classifier"]).__name__,
        **model_metadata(pipeline),
        "trained_at": trained_at,
        "saved": True,
        "train_rows": len(split.y_train),
        "test_rows": len(split.y_test),
        **metrics,
    }


@app.get("/model/status")
def model_status() -> dict:
    """Показать состояние модели, дату обучения (UTC) и метрики на test."""
    with training_lock:
        artifact = app.state.model_artifact
        return {
            "trained": app.state.churn_model is not None,
            "trained_at": artifact["trained_at"] if artifact else None,
            "metrics": artifact["metrics"] if artifact else None,
            "load_error": app.state.model_load_error,
            "model_type": artifact["model_type"] if artifact else None,
            "hyperparameters": artifact["hyperparameters"] if artifact else None,
        }


@app.get("/model/schema")
def model_schema() -> dict:
    """Описать входные признаки для /predict; доступно и до обучения модели."""
    schema = FeatureVectorChurn.model_json_schema()
    type_names = {"number": "float", "integer": "int", "string": "str"}
    return {
        "features": [
            {
                "name": name,
                "type": type_names[schema["properties"][name]["type"]],
                "group": "numeric" if name in NUMERIC_FEATURES else "categorical",
                "required": name in schema["required"],
            }
            for name in FEATURE_COLUMNS
        ],
        "feature_order": FEATURE_COLUMNS,
        "target": {"name": TARGET_COLUMN, "type": "int", "values": [0, 1], "included_in_request": False},
        "accepts_batch": True,
        "additional_features_allowed": False,
        "example": PREDICTION_EXAMPLE,
    }


@app.get("/model/metrics")
def model_metrics(
    limit: int = Query(default=5, ge=1, le=100),
    model_type: Literal["logreg", "random_forest"] | None = Query(default=None),
) -> dict:
    """Последнее успешное обучение и история от новых записей к старым.

    model_type фильтрует и latest, и history. limit ограничивает длину history.
    total — число записей после фильтрации до применения limit.
    История хранится в памяти процесса. После перезапуска из файла модели
    восстанавливается только последнее успешное обучение.
    При отсутствии обучений latest=null, history=[], total=0.
    """
    with training_lock:
        records = [record for record in reversed(app.state.training_history)
                   if model_type is None or record["model_type"] == model_type]
        return deepcopy({
            "latest": records[0] if records else None,
            "history": records[:limit],
            "total": len(records),
        })


# ----- Диагностика сервиса -----
@app.get("/health")
def health() -> dict:
    """Проверить модель в памяти и чтение непустого CSV.

    HTTP 200 означает, что сервис отвечает. status=degraded указывает,
    что модель или датасет пока недоступны. Проверка CSV не обучает модель.
    """
    dataset_error = None
    rows = []
    try:
        rows = load_dataset(DATASET_PATH)
        if not rows:
            dataset_error = "Датасет пуст."
    except FileNotFoundError:
        dataset_error = "Файл датасета не найден."
    except (DatasetError, OSError, UnicodeError):
        dataset_error = "Не удалось прочитать или проверить датасет."
    if dataset_error:
        logger.warning("health_dataset_unavailable reason=%s", dataset_error)
    model_available = app.state.churn_model is not None
    dataset_loaded = dataset_error is None
    return {
        "status": "ok" if model_available and dataset_loaded else "degraded",
        "model_available": model_available,
        "dataset_loaded": dataset_loaded,
        "dataset_rows": len(rows),
        "dataset_error": dataset_error,
    }
