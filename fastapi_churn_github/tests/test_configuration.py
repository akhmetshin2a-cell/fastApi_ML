"""Восстановление типа и параметров выбранной модели при старте."""
import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import joblib
import main


class ConfigurationTests(unittest.TestCase):
    def test_random_forest_settings_survive_restart_and_legacy_file_loads(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.joblib'
            with patch.object(main, 'MODEL_PATH', path):
                result = main.model_train(main.TrainingConfigChurn(
                    model_type='random_forest', hyperparameters={'n_estimators': 20, 'max_depth': 4}))
                async def restart():
                    async with main.lifespan(main.app):
                        status = main.model_status()
                        self.assertEqual(status['model_type'], 'random_forest')
                        self.assertEqual(status['hyperparameters']['n_estimators'], 20)
                        self.assertEqual(status['hyperparameters']['max_depth'], 4)
                        self.assertEqual(status['metrics']['f1'], result['f1'])
                asyncio.run(restart())
                # Файлы прежнего формата не содержат явных model_type/hyperparameters.
                artifact = joblib.load(path)
                artifact.pop('model_type')
                artifact.pop('hyperparameters')
                joblib.dump(artifact, path)
                restored = main.load_churn_model()
                self.assertEqual(restored['model_type'], 'random_forest')
                self.assertEqual(restored['hyperparameters']['max_depth'], 4)
