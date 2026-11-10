"""История успешных обучений и восстановление последней записи."""
import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from sklearn.metrics import roc_auc_score
import main


class HistoryTests(unittest.TestCase):
    def test_history_snapshots_filters_auc_and_failed_save(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(main, 'MODEL_PATH', Path(folder) / 'model.joblib'), patch.object(main.app.state, 'training_history', []):
                self.assertEqual(main.model_metrics(5, None), {'latest': None, 'history': [], 'total': 0})
                configurations = [main.TrainingConfigChurn(), main.TrainingConfigChurn(
                    model_type='random_forest', hyperparameters={'n_estimators': 20, 'max_depth': 4})]
                results = [main.model_train(config) for config in configurations]
                history = main.model_metrics(1, None)
                self.assertEqual(history['total'], 2)
                self.assertEqual(len(history['history']), 1)
                self.assertEqual(history['latest']['model_type'], 'random_forest')
                filtered = main.model_metrics(5, 'logreg')
                self.assertEqual(filtered['total'], 1)
                self.assertEqual(filtered['latest']['timestamp'], results[0]['trained_at'])
                split = main.prepare_data(main.load_dataset(), impute=False)
                model = main.app.state.churn_model
                expected = roc_auc_score(split.y_test, model.predict_proba(split.X_test)[:, list(model.classes_).index(1)])
                self.assertEqual(history['latest']['metrics']['roc_auc'], expected)
                history['latest']['metrics']['f1'] = -1
                self.assertGreaterEqual(main.model_metrics(1, None)['latest']['metrics']['f1'], 0)
                before = main.model_metrics(100, None)
                with patch.object(main.joblib, 'dump', side_effect=OSError('failed')):
                    with self.assertLogs(main.logger, level='ERROR'), self.assertRaises(HTTPException):
                        main.model_train()
                self.assertEqual(main.model_metrics(100, None), before)
                async def restart():
                    async with main.lifespan(main.app):
                        restored = main.model_metrics(100, None)
                        self.assertEqual(restored['total'], 1)
                        self.assertEqual(restored['latest'], before['latest'])
                        self.assertEqual(main.model_metrics(5, 'logreg')['total'], 0)
                asyncio.run(restart())
