"""Trees, forests, boosting, and the kernel SVM."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ml_from_scratch.kernel_methods import SVM
from ml_from_scratch.tree_models import (
    DecisionTreeClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    entropy,
    gini_impurity,
    information_gain,
    leaf_weight,
    second_order_gain,
)


class TreeTests(unittest.TestCase):
    def test_impurity_baselines(self) -> None:
        self.assertEqual(gini_impurity([1.0, 0.0]), 0.0)
        self.assertAlmostEqual(gini_impurity([1.0, 1.0]), 0.5)
        self.assertAlmostEqual(entropy([1.0, 1.0]), 1.0)
        self.assertAlmostEqual(information_gain([2, 2], [2, 0], [0, 2], "gini"), 0.5)
        self.assertAlmostEqual(information_gain([2, 2], [2, 0], [0, 2], "entropy"), 1.0)

    def test_threshold_split(self) -> None:
        features = np.array([[1.0], [2.0], [3.0], [4.0]])
        labels = np.array([0, 0, 1, 1])
        model = DecisionTreeClassifier().fit(features, labels)
        assert model.tree_ is not None
        self.assertIsNotNone(model.tree_.threshold)
        self.assertAlmostEqual(float(model.tree_.threshold), 2.5)
        self.assertEqual(model.predict(features).shape, (4,))
        self.assertEqual(model.score(features, labels), 1.0)
        probabilities = model.predict_proba(features)
        self.assertEqual(probabilities.shape, (4, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_one_unsampled_tree_matches_the_cart_fit(self) -> None:
        features = np.array([[1.0], [2.0], [3.0], [4.0], [1.5], [3.5]])
        labels = np.array([0, 0, 1, 1, 0, 1])
        forest = RandomForestClassifier(
            n_estimators=1,
            bootstrap=False,
            max_features=None,
            random_state=0,
        ).fit(features, labels)
        tree = DecisionTreeClassifier().fit(features, labels)
        np.testing.assert_array_equal(forest.predict(features), tree.predict(features))
        self.assertEqual(forest.predict_proba(features).shape, (6, 2))

    def test_second_order_leaf_and_stump(self) -> None:
        self.assertAlmostEqual(leaf_weight(1.0, 2.0, 0.0), -0.5)
        self.assertAlmostEqual(
            second_order_gain(1.0, 2.0, -1.0, 2.0, reg_lambda=0.0, gamma=0.0),
            0.5,
        )
        features = np.array([[1.0], [2.0], [3.0], [4.0]])
        targets = np.array([0.0, 0.0, 1.0, 1.0])
        model = GradientBoostingRegressor(
            n_estimators=1,
            learning_rate=1.0,
            max_depth=1,
            reg_lambda=0.0,
            gamma=0.0,
        ).fit(features, targets)
        np.testing.assert_allclose(model.predict(features), targets)
        self.assertEqual(model.predict(features).shape, (4,))
        self.assertGreater(model.loss_history_[0], model.loss_history_[-1])


class SupportVectorMachineTests(unittest.TestCase):
    def test_linear_svm_separates_and_obeys_the_dual_constraint(self) -> None:
        rng = np.random.default_rng(0)
        positive = rng.normal(loc=[2.0, 2.0], scale=0.3, size=(16, 2))
        negative = rng.normal(loc=[-2.0, -2.0], scale=0.3, size=(16, 2))
        features = np.vstack([positive, negative])
        labels = np.array([1] * 16 + [0] * 16)
        model = SVM(C=1.0, kernel="linear", tol=1e-3, max_iter=40).fit(features, labels)
        self.assertEqual(model.score(features, labels), 1.0)
        self.assertEqual(model.decision_function(features).shape, (32,))
        assert model.alpha_ is not None
        signed = np.where(labels == 1, 1.0, -1.0)
        self.assertAlmostEqual(float(np.sum(model.alpha_ * signed)), 0.0, places=6)
        self.assertTrue(np.all(model.alpha_ >= -1e-8))
        self.assertTrue(np.all(model.alpha_ <= 1.0 + 1e-8))

    def test_rbf_kernel_fits_xor(self) -> None:
        features = np.array(
            [
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-0.7, -0.7],
                [-0.7, 0.7],
                [0.7, -0.7],
                [0.7, 0.7],
            ]
        )
        labels = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        model = SVM(C=5.0, kernel="rbf", gamma=1.0, tol=1e-3, max_iter=80).fit(features, labels)
        self.assertEqual(model.score(features, labels), 1.0)
        assert model.support_vectors_ is not None
        self.assertEqual(model.support_vectors_.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
