"""Loss values, analytic derivatives, and optimizer steps."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ml_from_scratch.optimization.losses import (
    BinaryCrossEntropy,
    HingeLoss,
    MeanSquaredError,
    SoftmaxCrossEntropy,
    log_softmax,
    sigmoid,
    softmax,
)
from ml_from_scratch.optimization.optimizers import (
    Adam,
    SGD,
    finite_difference_gradient,
    gradient_check,
)


class LossBaselineTests(unittest.TestCase):
    def test_mean_squared_error_value_and_gradient(self) -> None:
        predicted = np.array([1.0, 1.0, 5.0])
        observed = np.array([1.0, 2.0, 4.0])
        # (1/(2m)) * (0 + 1 + 1) = 1/3.
        self.assertAlmostEqual(MeanSquaredError.loss(predicted, observed), 1.0 / 3.0)
        gradient = MeanSquaredError.gradient(predicted, observed)
        np.testing.assert_allclose(gradient, np.array([0.0, -1.0 / 3.0, 1.0 / 3.0]))
        self.assertEqual(gradient.shape, predicted.shape)

    def test_mean_squared_error_matches_central_difference(self) -> None:
        rng = np.random.default_rng(0)
        predicted = rng.normal(size=(5, 2))
        observed = rng.normal(size=(5, 2))
        analytic = MeanSquaredError.gradient(predicted, observed)
        numeric = finite_difference_gradient(
            lambda guess: MeanSquaredError.loss(guess, observed),
            predicted,
        )
        self.assertLess(gradient_check(analytic, numeric), 1e-7)

    def test_bernoulli_logit_baseline(self) -> None:
        logits = np.array([0.0, 0.0])
        labels = np.array([1.0, 0.0])
        self.assertAlmostEqual(BinaryCrossEntropy.loss_with_logits(logits, labels), np.log(2.0))
        gradient = BinaryCrossEntropy.gradient_with_logits(logits, labels)
        np.testing.assert_allclose(gradient, np.array([-0.25, 0.25]))
        self.assertEqual(gradient.shape, logits.shape)

    def test_bernoulli_logit_gradient_matches_central_difference(self) -> None:
        rng = np.random.default_rng(1)
        logits = rng.normal(size=6)
        labels = np.array([0.0, 1.0, 1.0, 0.0, 0.2, 0.8])
        analytic = BinaryCrossEntropy.gradient_with_logits(logits, labels)
        numeric = finite_difference_gradient(
            lambda scores: BinaryCrossEntropy.loss_with_logits(scores, labels),
            logits,
        )
        self.assertLess(gradient_check(analytic, numeric), 1e-7)

    def test_softmax_baseline_and_shape(self) -> None:
        logits = np.array([[0.0, 0.0]])
        probabilities = softmax(logits)
        np.testing.assert_allclose(probabilities, np.array([[0.5, 0.5]]))
        self.assertEqual(probabilities.shape, (1, 2))
        np.testing.assert_allclose(np.log(probabilities), log_softmax(logits))
        self.assertAlmostEqual(SoftmaxCrossEntropy.loss(logits, np.array([0])), np.log(2.0))
        gradient = SoftmaxCrossEntropy.gradient(logits, np.array([0]))
        np.testing.assert_allclose(gradient, np.array([[-0.5, 0.5]]))
        self.assertEqual(gradient.shape, logits.shape)

    def test_softmax_gradient_matches_central_difference(self) -> None:
        rng = np.random.default_rng(2)
        logits = rng.normal(size=(7, 4))
        labels = rng.integers(0, 4, size=7)
        analytic = SoftmaxCrossEntropy.gradient(logits, labels)
        numeric = finite_difference_gradient(
            lambda scores: SoftmaxCrossEntropy.loss(scores, labels),
            logits,
        )
        self.assertLess(gradient_check(analytic, numeric), 1e-7)
        one_hot = np.eye(4)[labels]
        np.testing.assert_allclose(
            SoftmaxCrossEntropy.gradient(logits, one_hot),
            analytic,
        )

    def test_hinge_baseline(self) -> None:
        scores = np.array([0.0, 3.0])
        labels = np.array([1.0, 1.0])
        self.assertAlmostEqual(HingeLoss.loss(scores, labels), 0.5)
        np.testing.assert_allclose(HingeLoss.gradient(scores, labels), np.array([-0.5, 0.0]))

    def test_sigmoid_fixed_points(self) -> None:
        self.assertAlmostEqual(float(sigmoid(0.0)), 0.5)
        self.assertAlmostEqual(float(sigmoid(40.0)), 1.0)
        self.assertAlmostEqual(float(sigmoid(-40.0)), 0.0)


class OptimizerStepTests(unittest.TestCase):
    def test_sgd_step(self) -> None:
        parameter = np.array([1.0, -2.0])
        SGD(learning_rate=0.1).step(parameter, np.array([0.5, 0.25]))
        np.testing.assert_allclose(parameter, np.array([0.95, -2.025]))

    def test_adam_first_step_is_unbiased(self) -> None:
        parameter = np.array([1.0, 0.0])
        Adam(learning_rate=0.1).step(parameter, np.array([1.0, 0.0]))
        expected = 1.0 - 0.1 / (1.0 + 1e-8)
        self.assertAlmostEqual(parameter[0], expected, places=10)
        self.assertEqual(parameter[1], 0.0)


if __name__ == "__main__":
    unittest.main()
