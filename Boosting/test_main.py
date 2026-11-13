"""ทดสอบด้วย python -m unittest -v"""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from main import ManualAdaBoost


class BoostingTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(6).reshape(-1, 1)
        self.y = np.array(["a", "a", "b", "b", "c", "c"])

    def test_error_alpha_and_weight_update(self):
        model = ManualAdaBoost(2).fit(self.X, self.y)
        wrong = model.models_[0].predict(self.X) != self.y
        # ต้นไม้แบ่งครั้งเดียว: ผิด 2 จาก 6, alpha=log(4)
        self.assertEqual(wrong.sum(), 2)
        self.assertAlmostEqual(model.history_[0][1], 1 / 3)
        self.assertAlmostEqual(model.alphas_[0], np.log(4))
        np.testing.assert_allclose(model.sample_weights_[0], np.full(6, 1 / 6))
        # หลัง normalize: ตัวที่ผิดหนัก 1/3, ตัวที่ถูกหนัก 1/12
        np.testing.assert_allclose(model.sample_weights_[1], np.where(wrong, 1 / 3, 1 / 12))
        wrong_next = model.models_[1].predict(self.X) != self.y
        self.assertAlmostEqual(model.history_[1][1], model.sample_weights_[1][wrong_next].sum())
        for weights in model.sample_weights_:
            self.assertAlmostEqual(weights.sum(), 1)
            self.assertTrue(np.isfinite(weights).all())
        self.assertFalse(np.shares_memory(model.sample_weights_[0], model.sample_weights_[1]))

    def test_weighted_voting(self):
        # สองต้นโหวต a แต่ต้นที่โหวต b มีน้ำหนักมากกว่า จึงต้องได้ b
        model = ManualAdaBoost()
        model.classes_ = np.array(["a", "b", "c"])
        model.models_ = [Mock(), Mock(), Mock()]
        model.alphas_ = [1, 1, 3]
        for tree, prediction in zip(model.models_, [["a", "c"], ["a", "b"], ["b", "c"]]):
            tree.predict.return_value = np.array(prediction)
        np.testing.assert_array_equal(model.predict([[0], [1]]), ["b", "c"])

    def test_training_error_history(self):
        model = ManualAdaBoost(5).fit(self.X, self.y)
        votes = np.zeros((6, 3))
        for tree, alpha, row in zip(model.models_, model.alphas_, model.history_):
            for i, label in enumerate(tree.predict(self.X)):
                votes[i, np.flatnonzero(model.classes_ == label)[0]] += alpha
            expected = np.mean(model.classes_[votes.argmax(axis=1)] != self.y)
            self.assertAlmostEqual(row[3], expected)
        self.assertEqual(len(model.sample_weights_), len(model.models_) + 1)

    def test_perfect_learner_keeps_previous_models(self):
        # จำลองรอบที่สองให้ถูกทั้งหมด เพื่อทดสอบทางเดิน error=0 หลังมีโมเดลแล้ว
        first, perfect = Mock(), Mock()
        first.predict.return_value = np.array(["a", "a", "b", "b", "b", "b"])
        perfect.predict.return_value = self.y.copy()
        with patch("main.DecisionTreeClassifier", side_effect=[first, perfect]):
            model = ManualAdaBoost(5).fit(self.X, self.y)
        self.assertEqual(len(model.models_), 2)
        self.assertEqual(model.history_[-1][1], 0)
        self.assertGreater(model.alphas_[-1], sum(model.alphas_[:-1]))
        self.assertTrue(np.isfinite(model.alphas_).all())
        np.testing.assert_array_equal(model.predict(self.X), self.y)
        np.testing.assert_array_equal(model.sample_weights_[-1], model.sample_weights_[-2])

    def test_reject_chance_learner(self):
        with self.assertRaises(ValueError):
            ManualAdaBoost().fit(np.zeros((6, 1)), self.y)

    def test_real_perfect_learner_and_predict_before_fit(self):
        with self.assertRaises(ValueError):
            ManualAdaBoost().predict(self.X)
        labels = np.array(["a", "a", "a", "b", "b", "b"])
        model = ManualAdaBoost().fit(self.X, labels)
        self.assertEqual(len(model.models_), 1)
        np.testing.assert_array_equal(model.predict(self.X), labels)


if __name__ == "__main__":
    unittest.main()
