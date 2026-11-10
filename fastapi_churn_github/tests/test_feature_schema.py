"""Схема признаков и сохранение всей предобработки для обоих классификаторов."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from pydantic import ValidationError
import main


class FeatureSchemaTests(unittest.TestCase):
    def test_schema_matches_pydantic_and_features(self):
        schema = main.model_schema()
        self.assertEqual(schema['feature_order'], main.FEATURE_COLUMNS)
        self.assertEqual(set(main.FEATURE_COLUMNS), set(main.FeatureVectorChurn.model_fields))
        self.assertEqual(len(schema['features']), 9)
        for field in schema['features']:
            annotation = main.FeatureVectorChurn.model_fields[field['name']].annotation
            self.assertEqual(field['type'], annotation.__name__)
            self.assertTrue(field['required'])
        self.assertNotIn('churn', schema['feature_order'])
        with self.assertRaises(ValidationError):
            main.FeatureVectorChurn(**{**main.PREDICTION_EXAMPLE, 'unknown': 5})

    def test_reordered_keys_give_identical_predictions_after_reload(self):
        rows = main.load_dataset()
        split = main.prepare_data(rows, impute=False)
        training = split.X_train.assign(churn=split.y_train).to_dict(orient='records')
        first = main.build_feature_frame([main.PREDICTION_EXAMPLE])
        reversed_keys = dict(reversed(list(main.PREDICTION_EXAMPLE.items())))
        second = main.build_feature_frame([reversed_keys])
        pd.testing.assert_frame_equal(first, second)
        for kind in ('logreg', 'random_forest'):
            config = main.TrainingConfigChurn(model_type=kind,
                hyperparameters={'n_estimators': 15} if kind == 'random_forest' else {})
            model = main.train_churn_model(training, config)
            self.assertEqual(list(model.feature_names_in_), main.FEATURE_COLUMNS)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'model.joblib'
                main.save_churn_model(model, '2026-11-10T13:00:00+00:00', {'accuracy': .8, 'f1': .2}, path)
                restored = main.load_churn_model(path)['pipeline']
                np.testing.assert_allclose(model.predict_proba(first), restored.predict_proba(second))
                numeric = restored.named_steps['preprocessor'].named_transformers_['numeric']
                expected_steps = {'imputer', 'scaler'} if kind == 'logreg' else {'imputer'}
                self.assertEqual(set(numeric.named_steps), expected_steps)
                categorical = restored.named_steps['preprocessor'].named_transformers_['categorical']
                self.assertEqual(set(categorical.named_steps), {'imputer', 'encoder'})

    def test_mismatched_saved_schema_rejected(self):
        rows = main.load_dataset()
        model = main.train_churn_model(rows)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.joblib'
            main.save_churn_model(model, '2026-11-10T13:00:00+00:00', {'accuracy': .8, 'f1': .2}, path)
            with patch.object(main, 'FEATURE_COLUMNS', list(reversed(main.FEATURE_COLUMNS))):
                with self.assertRaisesRegex(ValueError, 'Схема сохранённой'):
                    main.load_churn_model(path)
