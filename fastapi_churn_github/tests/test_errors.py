"""HTTP-проверки единого формата ошибок без технических трассировок."""
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
import urllib.error
import main


class ErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.csv_path = Path(cls.folder.name) / 'dataset.csv'
        model_path = Path(cls.folder.name) / 'model.joblib'
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        cls.base = f'http://127.0.0.1:{port}'
        script = f'''
from pathlib import Path
import main, uvicorn
main.DATASET_PATH = Path({str(cls.csv_path)!r})
main.MODEL_PATH = Path({str(model_path)!r})
class BrokenModel:
    def predict(self, frame):
        raise RuntimeError("PRIVATE_TRACE_MARKER")
@main.app.get('/testing/break-model')
def break_model():
    main.app.state.churn_model = BrokenModel()
    return {{"ok": True}}
@main.app.get('/testing/unexpected')
def unexpected():
    raise RuntimeError("PRIVATE_TRACE_MARKER")
@main.app.get('/testing/dataset-error')
def dataset_error():
    raise main.DatasetError("Неверный набор признаков.")
uvicorn.run(main.app, host='127.0.0.1', port={port}, log_level='critical')
'''
        cls.process = subprocess.Popen(
            [sys.executable, '-c', script], cwd=Path(main.__file__).resolve().parent,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(150):
            try:
                cls.call('/model/status')
                return
            except OSError:
                time.sleep(.1)
        cls.process.terminate()
        cls.process.wait(timeout=10)
        cls.folder.cleanup()
        raise RuntimeError('Server did not start')

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        cls.process.wait(timeout=10)
        cls.folder.cleanup()

    @classmethod
    def call(cls, path, data=None, method='GET', raw=None):
        body = raw if raw is not None else json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(cls.base + path, data=body, method=method,
                                     headers={'Content-Type': 'application/json'})
        try:
            response = urllib.request.urlopen(req, timeout=15)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)

    def assert_error(self, result, status, code):
        actual_status, body = result
        self.assertEqual(actual_status, status)
        self.assertEqual(set(body), {'code', 'message', 'details'})
        self.assertEqual(body['code'], code)
        self.assertTrue(body['message'])
        self.assertNotIn('Traceback', json.dumps(body))
        self.assertNotIn('PRIVATE_TRACE_MARKER', json.dumps(body))

    def test_request_validation_and_http_errors(self):
        for data in ({}, {**main.PREDICTION_EXAMPLE, 'extra': 1},
                     {**main.PREDICTION_EXAMPLE, 'usage_hours': 'abc'},
                     [{k: v for k, v in main.PREDICTION_EXAMPLE.items() if k != 'region'}]):
            result = self.call('/predict', data, 'POST')
            self.assert_error(result, 422, 'VALIDATION_ERROR')
            self.assertIsInstance(result[1]['details'], list)
        self.assert_error(self.call('/predict', method='POST', raw=b'{broken'), 422, 'VALIDATION_ERROR')
        self.assert_error(self.call('/missing'), 404, 'NOT_FOUND')
        self.assert_error(self.call('/predict'), 405, 'METHOD_NOT_ALLOWED')
        self.assert_error(self.call('/model/train', {'model_type': 'other'}, 'POST'), 422, 'VALIDATION_ERROR')

    def test_model_and_dataset_errors(self):
        self.assert_error(self.call('/predict', main.PREDICTION_EXAMPLE, 'POST'), 503, 'MODEL_NOT_READY')
        self.assert_error(self.call('/model/train', method='POST'), 404, 'DATASET_NOT_FOUND')
        self.csv_path.write_text(','.join(main.DatasetRowChurn.model_fields) + '\n')
        self.assert_error(self.call('/model/train', method='POST'), 422, 'EMPTY_DATASET')
        shutil.copyfile(main.DATASET_PATH, self.csv_path)
        self.assert_error(self.call('/model/train', {'hyperparameters': {'C': -1}}, 'POST'), 422, 'TRAINING_CONFIG_ERROR')
        self.assertEqual(self.call('/model/train', method='POST')[0], 200)
        self.assert_error(self.call('/predict', [], 'POST'), 422, 'EMPTY_INPUT')
        self.call('/testing/break-model')
        self.assert_error(self.call('/predict', main.PREDICTION_EXAMPLE, 'POST'), 500, 'PREDICTION_ERROR')
        self.assert_error(self.call('/testing/dataset-error'), 422, 'DATASET_ERROR')
        self.assert_error(self.call('/testing/unexpected'), 500, 'INTERNAL_ERROR')
        self.assertEqual(self.call('/model/status')[0], 200)

    def test_error_documentation(self):
        status, spec = self.call('/openapi.json')
        self.assertEqual(status, 200)
        for path in ('/predict', '/model/train'):
            responses = spec['paths'][path]['post']['responses']
            for code in ('422', '500'):
                content = responses[code]['content']['application/json']
                self.assertTrue(content['schema']['$ref'].endswith('/ErrorResponseChurn'))
                self.assertIn('examples', content)
