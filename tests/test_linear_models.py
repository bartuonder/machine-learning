"""Ordinary least squares, logistic regression, and softmax regression."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ml_from_scratch.linear_models import LinearRegression, LogisticRegression, SoftmaxRegression
from ml_from_scratch.optimization.optimizers import finite_difference_gradient, gradient_check
from ml_from_scratch.utils.data_manipulation import add_intercept


class OrdinaryLeastSquaresTests(unittest.TestCase):
    def test_normal_equation_recovers_a_line(self) -> None:
        features = np.array([[1.0], [2.0], [3.0]])
        targets = np.array([3.0, 5.0, 7.0])
        model = LinearRegression(method="normal_equation").fit(features, targets)
        self.assertAlmostEqual(model.intercept_, 1.0, places=8)
        np.testing.assert_allclose(model.coef_, [2.0])
        self.assertEqual(model.predict(features).shape, (3,))
        self.assertAlmostEqual(model.score(features, targets), 1.0)

    def test_gradient_descent_matches_normal_equation(self) -> None:
        rng = np.random.default_rng(0)
        features = rng.normal(size=(40, 3))
        features = (features - features.mean(axis=0)) / features.std(axis=0)
        targets = features @ np.array([0.5, -1.0, 0.25]) + 1.5
        closed = LinearRegression(method="normal_equation", l2=0.4).fit(features, targets)
        iterative = LinearRegression(
            method="gradient_descent",
            learning_rate=0.2,
            max_iter=2000,
            tol=1e-12,
            l2=0.4,
        ).fit(features, targets)
        np.testing.assert_allclose(iterative.coef_, closed.coef_, atol=1e-5)
        self.assertAlmostEqual(iterative.intercept_, closed.intercept_, places=5)
        self.assertEqual(closed.predict(features).shape, targets.shape)
        self.assertGreater(len(iterative.loss_history_), 1)

    def test_ols_gradient_matches_central_difference(self) -> None:
        rng = np.random.default_rng(1)
        features = rng.normal(size=(12, 2))
        targets = rng.normal(size=12)
        model = LinearRegression(l2=0.5, fit_intercept=True)
        design = add_intercept(features)
        theta = rng.normal(size=design.shape[1])
        analytic = model._gradient(design, targets, theta, features.shape[0])
        numeric = finite_difference_gradient(
            lambda point: model._objective(design, targets, point),
            theta,
        )
        self.assertLess(gradient_check(analytic, numeric), 1e-7)
        self.assertEqual(analytic.shape, theta.shape)


class LogisticRegressionTests(unittest.TestCase):
    def test_bernoulli_mle_for_an_intercept(self) -> None:
        features = np.ones((3, 1))
        labels = np.array([1, 1, 0])
        model = LogisticRegression(
            method="newton",
            fit_intercept=False,
            tol=1e-12,
        ).fit(features, labels)
        np.testing.assert_allclose(model.coef_, [np.log(2.0)], atol=1e-8)
        probabilities = model.predict_proba(features)
        self.assertEqual(probabilities.shape, (3, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        np.testing.assert_allclose(probabilities[:, 1], 2.0 / 3.0, atol=1e-8)

    def test_logistic_gradient_and_hessian_match_central_differences(self) -> None:
        rng = np.random.default_rng(2)
        features = rng.normal(size=(16, 3))
        labels = (rng.random(16) > 0.4).astype(float)
        model = LogisticRegression(l2=0.7, fit_intercept=True)
        design = add_intercept(features)
        theta = rng.normal(scale=0.3, size=design.shape[1])
        analytic = model._gradient(design, labels, theta, features.shape[0])
        numeric = finite_difference_gradient(
            lambda point: model._objective(design, labels, point),
            theta,
        )
        self.assertLess(gradient_check(analytic, numeric), 1e-7)
        hessian = model._hessian(design, labels, theta, features.shape[0])
        self.assertEqual(hessian.shape, (theta.size, theta.size))
        numeric_hessian = np.empty_like(hessian)
        step = 1e-5
        for coordinate in range(theta.size):
            direction = np.zeros_like(theta)
            direction[coordinate] = step
            plus = model._gradient(design, labels, theta + direction, features.shape[0])
            minus = model._gradient(design, labels, theta - direction, features.shape[0])
            numeric_hessian[:, coordinate] = (plus - minus) / (2.0 * step)
        self.assertLess(gradient_check(hessian, numeric_hessian), 1e-5)

    def test_gradient_descent_matches_newton(self) -> None:
        rng = np.random.default_rng(3)
        features = rng.normal(size=(60, 2))
        features = (features - features.mean(axis=0)) / features.std(axis=0)
        labels = (features @ np.array([0.8, -0.4]) + rng.normal(scale=0.5, size=60) > 0).astype(int)
        newton = LogisticRegression(method="newton", l2=1.0, tol=1e-10).fit(features, labels)
        gradient = LogisticRegression(
            method="gradient_descent",
            l2=1.0,
            learning_rate=0.4,
            max_iter=1500,
            tol=1e-12,
        ).fit(features, labels)
        np.testing.assert_allclose(gradient.coef_, newton.coef_, atol=1e-4)
        self.assertAlmostEqual(gradient.intercept_, newton.intercept_, places=4)
        self.assertEqual(newton.decision_function(features).shape, (60,))


class SoftmaxRegressionTests(unittest.TestCase):
    def test_softmax_parameter_gradient_matches_central_difference(self) -> None:
        rng = np.random.default_rng(4)
        features = rng.normal(size=(18, 2))
        labels = rng.integers(0, 3, size=18)
        model = SoftmaxRegression(l2=0.2, fit_intercept=True)
        design = add_intercept(features)
        theta = rng.normal(scale=0.2, size=(design.shape[1], 3))
        analytic = model._gradient(design, labels, theta, features.shape[0])
        numeric = finite_difference_gradient(
            lambda point: model._objective(design, labels, point),
            theta,
        )
        self.assertEqual(analytic.shape, theta.shape)
        self.assertLess(gradient_check(analytic, numeric), 1e-6)

    def test_softmax_forward_shapes_and_a_separable_problem(self) -> None:
        rng = np.random.default_rng(5)
        centers = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.2]])
        features = np.vstack(
            [rng.normal(loc=center, scale=0.2, size=(25, 2)) for center in centers]
        )
        labels = np.repeat([0, 1, 2], 25)
        model = SoftmaxRegression(learning_rate=0.5, max_iter=300, tol=1e-8).fit(features, labels)
        probabilities = model.predict_proba(features)
        scores = model.decision_function(features)
        self.assertEqual(scores.shape, (75, 3))
        self.assertEqual(probabilities.shape, (75, 3))
        self.assertEqual(model.coef_.shape, (3, 2))
        self.assertEqual(model.intercept_.shape, (3,))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertEqual(model.score(features, labels), 1.0)


if __name__ == "__main__":
    unittest.main()
