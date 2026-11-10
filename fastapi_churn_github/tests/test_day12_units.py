"""Проверки функций ML без HTTP и без запуска FastAPI."""
import numpy as np
import pandas as pd
import pytest
import main


def test_read_csv_and_feature_types(csv_path, churn_rows):
    loaded = main.load_dataset(csv_path)
    assert loaded == churn_rows
    assert isinstance(loaded[0]['monthly_fee'], float)
    assert isinstance(loaded[0]['churn'], int)


def test_split_is_reproducible_and_separates_target(churn_rows):
    first = main.prepare_data(churn_rows, impute=False)
    second = main.prepare_data(churn_rows, impute=False)
    assert first.X_train.shape == (32, 9)
    assert first.X_test.shape == (8, 9)
    assert set(first.X_train.index).isdisjoint(first.X_test.index)
    assert len(set(first.X_train.index) | set(first.X_test.index)) == 40
    assert 'churn' not in first.X_train
    assert first.y_train.value_counts().to_dict() == {0: 16, 1: 16}
    assert first.y_test.value_counts().to_dict() == {0: 4, 1: 4}
    pd.testing.assert_frame_equal(first.X_train, second.X_train)


@pytest.mark.parametrize('model_type,params', [
    ('logreg', {'C': 1.0}),
    ('random_forest', {'n_estimators': 15, 'max_depth': 4, 'random_state': 42}),
])
def test_training_is_reproducible_and_predicts_heldout_clients(churn_rows, model_type, params):
    split = main.prepare_data(churn_rows, impute=False)
    training = split.X_train.assign(churn=split.y_train).to_dict(orient='records')
    config = main.TrainingConfigChurn(model_type=model_type, hyperparameters=params)
    first = main.train_churn_model(training, config)
    second = main.train_churn_model(training, config)
    np.testing.assert_allclose(first.predict_proba(split.X_test), second.predict_proba(split.X_test))
    # Синтетические классы намеренно хорошо разделимы по поведению.
    np.testing.assert_array_equal(first.predict(split.X_test), split.y_test)


def test_imputation_uses_train_statistics_only(churn_rows):
    split = main.prepare_data(churn_rows, impute=False)
    train_ids = list(split.X_train.index)
    test_ids = list(split.X_test.index)
    for i in train_ids:
        churn_rows[i]['monthly_fee'] = 10.0
        churn_rows[i]['region'] = 'europe'
    for i in test_ids:
        churn_rows[i]['monthly_fee'] = 999999.0
        churn_rows[i]['region'] = 'asia'
    for i in (train_ids[0], test_ids[0]):
        churn_rows[i]['monthly_fee'] = None
        churn_rows[i]['region'] = None
    prepared = main.prepare_data(churn_rows)
    for frame, index in ((prepared.X_train, train_ids[0]), (prepared.X_test, test_ids[0])):
        assert frame.loc[index, 'monthly_fee'] == 10
        assert frame.loc[index, 'region'] == 'europe'
        assert not frame.isna().any().any()


@pytest.mark.parametrize('rows', [[], [{'churn': 0}]])
def test_unusable_training_data_rejected(rows):
    with pytest.raises(main.DatasetError):
        main.train_churn_model(rows)


def test_split_rejects_test_without_both_classes(churn_rows):
    rows = [dict(row, churn=0 if i < 18 else 1) for i, row in enumerate(churn_rows[:20])]
    with pytest.raises(main.DatasetError, match='В train и test нужны оба класса'):
        main.prepare_data(rows, test_size=0.1, impute=False)
