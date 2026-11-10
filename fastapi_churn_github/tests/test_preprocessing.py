"""Проверки разбиения и отсутствия утечки статистик из test."""

import csv
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from main import load_dataset, DatasetError, prepare_data


class PreparationTests(unittest.TestCase):
    def test_split_is_reproducible_disjoint_and_stratified(self):
        rows = load_dataset()
        first = prepare_data(rows)
        second = prepare_data(rows)
        self.assertEqual(len(first.y_train), 1600)
        self.assertEqual(len(first.y_test), 400)
        self.assertFalse(set(first.X_train.index) & set(first.X_test.index))
        self.assertEqual(len(set(first.X_train.index) | set(first.X_test.index)), len(rows))
        pd.testing.assert_frame_equal(first.X_train, second.X_train)
        self.assertNotIn("churn", first.X_train.columns)
        self.assertLess(abs(first.y_train.mean() - first.y_test.mean()), 0.01)

    def test_csv_missing_values_and_train_only_imputation(self):
        rows = load_dataset()
        split = prepare_data(rows)
        train_ids = list(split.X_train.index)
        test_ids = list(split.X_test.index)
        for i in train_ids:
            rows[i]["monthly_fee"] = 10.0
            rows[i]["region"] = "europe"
        for i in test_ids:
            rows[i]["monthly_fee"] = 1000000.0
            rows[i]["region"] = "asia"
        for i in (train_ids[0], test_ids[0]):
            rows[i]["monthly_fee"] = None
            rows[i]["region"] = None
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.csv"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            prepared = prepare_data(load_dataset(path))
        for frame, index in ((prepared.X_train, train_ids[0]), (prepared.X_test, test_ids[0])):
            self.assertEqual(frame.loc[index, "monthly_fee"], 10.0)
            self.assertEqual(frame.loc[index, "region"], "europe")
            self.assertFalse(frame.isna().any().any())

    def test_unusable_data_has_clear_error(self):
        with self.assertRaises(DatasetError):
            prepare_data([])
        rows = load_dataset()
        for row in rows:
            row["usage_hours"] = None
        with self.assertRaisesRegex(DatasetError, "полностью пустые"):
            prepare_data(rows)
        with self.assertRaises(DatasetError):
            prepare_data([{**row, "churn": 0} for row in rows])


if __name__ == "__main__":
    unittest.main()
