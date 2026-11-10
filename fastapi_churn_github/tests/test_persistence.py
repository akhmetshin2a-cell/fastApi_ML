"""Проверки сохранения, старта приложения и ошибки записи."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
import main


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'models' / 'churn_model.joblib'
        patcher = patch.object(main, 'MODEL_PATH', self.path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_roundtrip_restores_predictions_and_status(self):
        result = main.model_train()
        sample = main.prepare_data(main.load_dataset(), impute=False).X_test
        expected = main.app.state.churn_model.predict_proba(sample)
        artifact = main.load_churn_model()
        np.testing.assert_allclose(artifact['pipeline'].predict_proba(sample), expected)
        async def restart():
            async with main.lifespan(main.app):
                status = main.model_status()
                self.assertTrue(status['trained'])
                self.assertEqual(status['trained_at'], result['trained_at'])
                self.assertEqual(status['metrics']['accuracy'], result['accuracy'])
                self.assertEqual(status['metrics']['f1'], result['f1'])
        asyncio.run(restart())

    def test_missing_file_starts_without_model(self):
        self.assertIsNone(main.load_churn_model())
        async def start():
            async with main.lifespan(main.app):
                self.assertEqual(main.model_status(), dict(trained=False, trained_at=None, metrics=None, load_error=None, model_type=None, hyperparameters=None))
        asyncio.run(start())

    def test_corrupt_file_reports_error_and_allows_retraining(self):
        self.path.parent.mkdir()
        self.path.write_bytes(b'corrupt data')
        async def start():
            with self.assertLogs(main.logger, level='ERROR'):
                async with main.lifespan(main.app):
                    self.assertFalse(main.model_status()['trained'])
                    self.assertIsNotNone(main.model_status()['load_error'])
                    main.model_train()
                    self.assertTrue(main.model_status()['trained'])
                    self.assertIsNone(main.model_status()['load_error'])
        asyncio.run(start())

    def test_failed_save_preserves_previous_file_and_memory(self):
        main.model_train()
        previous = self.path.read_bytes()
        previous_model = main.app.state.churn_model
        with patch.object(main.joblib, 'dump', side_effect=OSError('write failed')):
            with self.assertLogs(main.logger, level='ERROR'):
                with self.assertRaises(HTTPException) as error:
                    main.model_train()
        self.assertEqual(error.exception.status_code, 500)
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertIs(main.app.state.churn_model, previous_model)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
