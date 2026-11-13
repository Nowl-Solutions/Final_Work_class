"""Verify that the comparison uses the same data and the requested learners."""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from bagging import fit_bagging
from boosting import fit_boosting
from compare_models import evaluate, run_comparison
from iris_baseline import load_iris_split


class ComparisonTests(unittest.TestCase):
    def test_same_data_and_four_model_ensembles(self):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch("compare_models.load_iris_split", wraps=load_iris_split) as loader,
            patch("compare_models.fit_bagging", wraps=fit_bagging) as bagging,
            patch("compare_models.fit_boosting", wraps=fit_boosting) as boosting,
            patch("compare_models.evaluate", wraps=evaluate) as evaluator,
        ):
            report = run_comparison(n_jobs=1)

        loader.assert_called_once_with(random_state=42)
        self.assertIs(bagging.call_args.args[1], boosting.call_args.args[1])
        self.assertIs(bagging.call_args.args[2], boosting.call_args.args[2])
        self.assertEqual(bagging.call_args.args[1].shape, (120, 4))
        self.assertEqual(boosting.call_args.kwargs["n_rounds"], 4)
        self.assertEqual(list(bagging.call_args.args[0]), [
            "svm", "knn", "naive_bayes", "logistic_regression",
        ])
        self.assertEqual(
            [type(model).__name__ for model in bagging.call_args.args[0].values()],
            ["SVC", "KNeighborsClassifier", "GaussianNB", "LogisticRegression"],
        )

        test_labels = evaluator.call_args_list[0].args[2]
        for call in evaluator.call_args_list:
            self.assertIs(call.args[2], test_labels)
        np.testing.assert_array_equal(report["y_test"], test_labels)
        self.assertEqual(report["train_samples"], 120)
        self.assertEqual(report["test_samples"], 30)
        self.assertEqual([row["model_count"] for row in report["results"]], [1, 4, 4])
        self.assertEqual([row["method"] for row in report["results"]], [
            "Single SVM", "Bagging (4 types)", "Boosting (4 types)",
        ])
        for row in report["results"]:
            predictions = np.array(row["predictions"])
            self.assertEqual(len(predictions), 30)
            correct = int((predictions == test_labels).sum())
            self.assertEqual(row["correct"], correct)
            self.assertAlmostEqual(row["accuracy"], correct / 30)
            self.assertEqual(np.array(row["confusion_matrix"]).sum(), 30)


if __name__ == "__main__":
    unittest.main()
