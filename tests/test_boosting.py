"""Check the boosting math, edge cases, and the real Iris workflow."""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from base_models import make_classifiers
from boosting import MODEL_NAMES, fit_boosting, predict_boosting, train_one_round
from iris_baseline import load_iris_split


class FixedPredictions:
    def __init__(self, predictions):
        self.predictions = np.array(predictions)

    def predict(self, X):
        return self.predictions.copy()


def fit_quietly(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fit_boosting(*args, **kwargs)


class BoostingTests(unittest.TestCase):
    def test_matches_sklearn_samme_for_gaussian_nb(self):
        # The reference is used only for verification, not in boosting.py.
        X_train, X_test, y_train, _, _ = load_iris_split()
        models, weights, classes = fit_quietly(
            {"naive_bayes": GaussianNB()}, X_train, y_train, n_rounds=4,
        )

        scaler = StandardScaler().fit(X_train)
        reference = AdaBoostClassifier(
            estimator=GaussianNB(), n_estimators=4, random_state=42,
        )
        reference.fit(scaler.transform(X_train), y_train)

        self.assertEqual(len(models), len(reference.estimators_))
        np.testing.assert_allclose(weights, reference.estimator_weights_[:len(models)])
        np.testing.assert_array_equal(
            predict_boosting(models, weights, classes, X_test),
            reference.predict(scaler.transform(X_test)),
        )

    def test_mistake_gets_more_weight_in_next_round(self):
        X = np.arange(6).reshape(-1, 1)
        y = np.array([0, 0, 1, 1, 2, 2])
        weights_seen = []

        def fake_train(base_model, X_train, y_train, sample_weights, seed):
            weights_seen.append(sample_weights.copy())
            if len(weights_seen) == 1:
                return FixedPredictions([0, 1, 1, 1, 2, 2])
            return FixedPredictions(y)

        with patch("boosting.train_one_round", side_effect=fake_train):
            fit_quietly({"example": object()}, X, y, n_rounds=2)

        np.testing.assert_allclose(weights_seen[0], np.full(6, 1 / 6))
        # One mistake out of six with three classes gives it 2/3 next round.
        self.assertAlmostEqual(weights_seen[1][1], 2 / 3)
        self.assertAlmostEqual(weights_seen[1].sum(), 1.0)
        self.assertGreater(weights_seen[1][1], weights_seen[1][0])

    def test_perfect_round_stops_and_dominates_previous_votes(self):
        X = np.arange(6).reshape(-1, 1)
        y = np.array([0, 0, 1, 1, 2, 2])
        with patch("boosting.train_one_round", side_effect=[
            FixedPredictions([0, 1, 1, 1, 2, 2]), FixedPredictions(y),
        ]) as trainer:
            models, weights, classes = fit_quietly(
                {"example": object()}, X, y, n_rounds=10,
            )

        self.assertEqual(trainer.call_count, 2)
        self.assertTrue(np.isfinite(weights).all())
        self.assertGreater(weights[-1], weights[:-1].sum())
        np.testing.assert_array_equal(predict_boosting(models, weights, classes, X), y)

    def test_bad_round_is_skipped_without_changing_weights(self):
        X = np.arange(6).reshape(-1, 1)
        y = np.array([0, 0, 1, 1, 2, 2])
        with patch("boosting.train_one_round", side_effect=[
            FixedPredictions((y + 1) % 3), FixedPredictions(y),
        ]) as trainer:
            models, weights, _ = fit_quietly(
                {"bad": object(), "good": object()}, X, y, n_rounds=2,
            )
            np.testing.assert_allclose(trainer.call_args.args[3], np.full(6, 1 / 6))
        self.assertEqual(len(models), 1)
        self.assertEqual(len(weights), 1)

    def test_random_guessing_threshold_is_rejected(self):
        X = np.arange(6).reshape(-1, 1)
        y = np.array([0, 0, 1, 1, 2, 2])
        with patch("boosting.train_one_round", return_value=FixedPredictions([0] * 6)):
            with self.assertRaisesRegex(ValueError, "No model passed"):
                fit_quietly({"example": object()}, X, y, n_rounds=1)

    def test_knn_sampling_uses_probability_weights(self):
        X = np.array([[10.0], [20.0], [30.0]])
        y = np.array([0, 1, 2])
        base_model = KNeighborsClassifier(n_neighbors=1)
        learner = train_one_round(
            base_model, X, y, np.array([1.0, 0.0, 0.0]), seed=42,
        )

        np.testing.assert_array_equal(learner.named_steps["model"].classes_, [0])
        np.testing.assert_allclose(learner.named_steps["scale"].mean_, [10.0])
        self.assertEqual(learner.named_steps["scale"].n_samples_seen_, 3)
        self.assertFalse(hasattr(base_model, "classes_"))

    def test_weighted_vote_and_ties_with_string_labels(self):
        X = np.zeros((1, 1))
        classes = np.array(["a", "b"])
        models = [FixedPredictions(["a"]), FixedPredictions(["a"]), FixedPredictions(["b"])]
        np.testing.assert_array_equal(
            predict_boosting(models, [1.0, 1.0, 3.0], classes, X), ["b"],
        )
        np.testing.assert_array_equal(
            predict_boosting(models, [1.0, 1.0, 2.0], classes, X), ["a"],
        )

    def test_iris_model_order_and_reproducibility(self):
        X_train, X_test, y_train, _, _ = load_iris_split()
        available_models = make_classifiers()
        selected = {name: available_models[name] for name in MODEL_NAMES}
        first = fit_quietly(selected, X_train, y_train)
        second = fit_quietly(selected, X_train, y_train)

        actual_types = [type(model.named_steps["model"]).__name__ for model in first[0]]
        self.assertEqual(actual_types, [
            "SVC", "KNeighborsClassifier", "GaussianNB", "LogisticRegression",
        ])
        np.testing.assert_allclose(first[1], second[1])
        np.testing.assert_array_equal(
            predict_boosting(*first, X_test), predict_boosting(*second, X_test),
        )
        for model in selected.values():
            self.assertFalse(hasattr(model, "classes_"))


if __name__ == "__main__":
    unittest.main()
