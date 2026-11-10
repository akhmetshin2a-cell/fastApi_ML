"""Проверки pipeline и ошибок загрузки при обучении."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException
from sklearn.metrics import accuracy_score, f1_score

import main


class TrainingTests(unittest.TestCase):
    def test_pipeline_handles_missing_and_unknown_categories(self):
        rows = main.load_dataset()
        split = main.prepare_data(rows, impute=False)
        training = split.X_train.assign(churn=split.y_train).to_dict(orient='records')
        training[0]['monthly_fee'] = None
        training[0]['region'] = None
        model = main.train_churn_model(training)
        numeric = model.named_steps['preprocessor'].named_transformers_['numeric']
        expected = pd.DataFrame(training)['monthly_fee'].median()
        self.assertAlmostEqual(numeric.named_steps['imputer'].statistics_[0], expected)
        sample = split.X_test.iloc[:1].copy()
        sample['monthly_fee'] = float('nan')
        sample['region'] = 'unknown_region'
        prediction = model.predict(sample)
        self.assertIn(int(prediction[0]), (0, 1))
        self.assertAlmostEqual(float(model.predict_proba(sample).sum()), 1.0)

    def test_endpoint_metrics_match_heldout_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(main, "MODEL_PATH", Path(folder) / "model.joblib"):
                result = main.model_train()
        split = main.prepare_data(main.load_dataset(), impute=False)
        predictions = main.app.state.churn_model.predict(split.X_test)
        self.assertEqual(result['accuracy'], accuracy_score(split.y_test, predictions))
        self.assertEqual(result['f1'], f1_score(split.y_test, predictions))
        self.assertEqual(result['train_rows'], 1600)
        self.assertEqual(result['test_rows'], 400)

    def test_absent_and_empty_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'missing.csv'
            with patch.object(main, 'DATASET_PATH', path):
                with self.assertRaises(HTTPException) as context:
                    main.model_train()
                self.assertEqual(context.exception.status_code, 404)
                path.write_text(','.join(main.DatasetRowChurn.model_fields) + '\n')
                with self.assertRaises(HTTPException) as context:
                    main.model_train()
                self.assertEqual(context.exception.status_code, 422)
        with self.assertRaises(main.DatasetError):
            main.train_churn_model([])
