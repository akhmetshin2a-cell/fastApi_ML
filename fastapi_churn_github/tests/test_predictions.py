"""HTTP-проверки одиночных и пакетных предсказаний."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
import urllib.error

import pandas as pd
import main


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.model_path = Path(self.folder.name) / 'churn.joblib'
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.base = f'http://127.0.0.1:{port}'
        script = ('import main, uvicorn; from pathlib import Path; '
                  f'main.MODEL_PATH = Path({str(self.model_path)!r}); '
                  f'uvicorn.run(main.app, host="127.0.0.1", port={port}, log_level="error")')
        self.process = subprocess.Popen(
            [sys.executable, '-c', script], cwd=Path(main.__file__).resolve().parent,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            for _ in range(150):
                try:
                    self.request('/model/status')
                    break
                except OSError:
                    time.sleep(.1)
            else:
                raise RuntimeError('Server did not start')
            try:
                self.request('/predict', main.PREDICTION_EXAMPLE)
                raise AssertionError('Missing model accepted')
            except urllib.error.HTTPError as error:
                assert error.code == 503
                assert 'POST /model/train' in json.load(error)['message']
            self.request('/model/train', post=True)
            self.model = main.load_churn_model(self.model_path)['pipeline']
        except BaseException:
            self.process.terminate()
            self.process.wait(timeout=10)
            self.folder.cleanup()
            raise

    def tearDown(self):
        self.process.terminate()
        self.process.wait(timeout=10)
        self.folder.cleanup()

    def request(self, path, data=None, post=False):
        req = urllib.request.Request(self.base + path,
            data=json.dumps(data).encode() if data is not None else None,
            method='POST' if post or data is not None else 'GET',
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.load(response)

    def test_single_and_batch_match_model_and_preserve_order(self):
        clients = [main.PREDICTION_EXAMPLE, main.SECOND_PREDICTION_EXAMPLE,
                   {**main.PREDICTION_EXAMPLE, 'usage_hours': 100, 'account_age_months': 60}]
        batch = self.request('/predict', clients)
        frame = pd.DataFrame(clients, columns=main.FEATURE_COLUMNS)
        labels = self.model.predict(frame)
        probabilities = self.model.predict_proba(frame)
        for client, result, label, values in zip(clients, batch, labels, probabilities):
            self.assertEqual(self.request('/predict', client), result)
            self.assertEqual(result['churn'], int(label))
            self.assertEqual(set(result['probabilities']), {'0', '1'})
            self.assertAlmostEqual(sum(result['probabilities'].values()), 1)
            for cls, value in zip(self.model.classes_, values):
                self.assertAlmostEqual(result['probabilities'][str(int(cls))], value)
        self.assertIsInstance(self.request('/predict', [clients[0]]), list)

    def test_invalid_and_empty_requests(self):
        for data in ([], {}, [{**main.PREDICTION_EXAMPLE, 'usage_hours': 'abc'}]):
            with self.assertRaises(urllib.error.HTTPError) as context:
                self.request('/predict', data)
            self.assertEqual(context.exception.code, 422)

    def test_unknown_category_and_documentation(self):
        result = self.request('/predict', {**main.PREDICTION_EXAMPLE, 'region': 'unknown'})
        self.assertIn(result['churn'], (0, 1))
        spec = self.request('/openapi.json')
        operation = spec['paths']['/predict']['post']
        self.assertEqual(set(operation['requestBody']['content']['application/json']['examples']), {'single', 'batch'})
        self.assertIn('examples', operation['responses']['200']['content']['application/json'])
        self.assertIn('PredictionResponseChurn', spec['components']['schemas'])

    def test_z_training_configuration_and_invalid_parameters(self):
        for config in (
            {'model_type': 'logreg', 'hyperparameters': {'C': 0.5, 'max_iter': 500}},
            {'model_type': 'random_forest', 'hyperparameters': {'n_estimators': 30, 'max_depth': 5, 'random_state': 42}},
        ):
            response = self.request('/model/train', config)
            self.assertEqual(response['model_type'], config['model_type'])
            status = self.request('/model/status')
            self.assertEqual(status['model_type'], config['model_type'])
            for name, value in config['hyperparameters'].items():
                self.assertEqual(status['hyperparameters'][name], value)
            artifact = main.load_churn_model(self.model_path)
            self.assertEqual(artifact['model_type'], config['model_type'])
            self.assertEqual(artifact['hyperparameters'], status['hyperparameters'])
            sample = pd.DataFrame([main.PREDICTION_EXAMPLE], columns=main.FEATURE_COLUMNS)
            result = self.request('/predict', main.PREDICTION_EXAMPLE)
            self.assertEqual(result['churn'], int(artifact['pipeline'].predict(sample)[0]))
        before = self.request('/model/status')
        file_before = self.model_path.read_bytes()
        for config in (
            {'model_type': 'unknown', 'hyperparameters': {}},
            {'model_type': 'logreg', 'hyperparameters': {'n_estimators': 10}},
            {'model_type': 'logreg', 'hyperparameters': {'C': -1}},
            {'model_type': 'random_forest', 'hyperparameters': {'n_estimators': 0}},
            {'model_type': 'random_forest', 'hyperparameters': {'max_depth': 'abc'}},
            {'model_type': 'logreg', 'hyperparameters': []},
        ):
            with self.assertRaises(urllib.error.HTTPError) as context:
                self.request('/model/train', config)
            self.assertEqual(context.exception.code, 422)
        self.assertEqual(self.request('/model/status'), before)
        self.assertEqual(self.model_path.read_bytes(), file_before)

    def test_schema_endpoint_and_reordered_json_keys(self):
        schema = self.request('/model/schema')
        self.assertEqual(schema['feature_order'], main.FEATURE_COLUMNS)
        self.assertEqual({f['name'] for f in schema['features']}, set(main.FeatureVectorChurn.model_fields))
        self.assertEqual(self.request('/predict', main.PREDICTION_EXAMPLE),
                         self.request('/predict', dict(reversed(list(main.PREDICTION_EXAMPLE.items())))))
        for data in ({**main.PREDICTION_EXAMPLE, 'surprise': 1},
                     {k: v for k, v in main.PREDICTION_EXAMPLE.items() if k != 'region'}):
            with self.assertRaises(urllib.error.HTTPError) as context:
                self.request('/predict', data)
            self.assertEqual(context.exception.code, 422)
        self.assertIn('/model/schema', self.request('/openapi.json')['paths'])

    def test_y_metrics_history_and_filters(self):
        before = self.request('/model/metrics')['total']
        self.request('/model/train', {'model_type': 'random_forest', 'hyperparameters': {'n_estimators': 20, 'max_depth': 4}})
        self.request('/model/train', post=True)
        result = self.request('/model/metrics?limit=1')
        self.assertEqual(result['total'], before + 2)
        self.assertEqual(len(result['history']), 1)
        self.assertEqual(result['latest']['model_type'], 'logreg')
        self.assertEqual(result['latest']['metrics'], self.request('/model/status')['metrics'])
        filtered = self.request('/model/metrics?model_type=random_forest&limit=3')
        self.assertTrue(filtered['history'])
        self.assertTrue(all(r['model_type'] == 'random_forest' for r in filtered['history']))
        for path in ('/model/metrics?limit=0', '/model/metrics?limit=101', '/model/metrics?model_type=other'):
            with self.assertRaises(urllib.error.HTTPError) as context:
                self.request(path)
            self.assertEqual(context.exception.code, 422)
        self.assertIn('/model/metrics', self.request('/openapi.json')['paths'])
