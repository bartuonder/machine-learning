"""Backpropagation and forward shapes for the one-hidden-layer network."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ml_from_scratch.neural_networks import MLP
from ml_from_scratch.optimization.optimizers import finite_difference_gradient, gradient_check


class MLPTests(unittest.TestCase):
    def test_classification_backpropagation_matches_central_difference(self) -> None:
        rng = np.random.default_rng(0)
        features = rng.normal(size=(6, 3))
        labels = np.array([0, 1, 2, 0, 1, 2])
        model = MLP(
            hidden_units=4,
            task="classification",
            optimizer="sgd",
            learning_rate=0.05,
            max_iter=1,
            random_state=0,
            l2=0.1,
        ).fit(features, labels)
        codes = np.searchsorted(model.classes_, labels)
        analytic = _flat_gradient(model, features, codes)
        flat = _flat_parameters(model)
        numeric = finite_difference_gradient(
            lambda point: _objective_at(model, features, codes, point),
            flat,
        )
        self.assertLess(gradient_check(analytic, numeric), 1e-6)

    def test_regression_backpropagation_matches_central_difference(self) -> None:
        rng = np.random.default_rng(1)
        features = rng.normal(size=(7, 2))
        targets = rng.normal(size=7)
        model = MLP(
            hidden_units=3,
            task="regression",
            activation="sigmoid",
            optimizer="sgd",
            learning_rate=0.05,
            max_iter=1,
            random_state=1,
            l2=0.05,
        ).fit(features, targets)
        analytic = _flat_gradient(model, features, targets)
        flat = _flat_parameters(model)
        numeric = finite_difference_gradient(
            lambda point: _objective_at(model, features, targets, point),
            flat,
        )
        self.assertLess(gradient_check(analytic, numeric), 1e-6)

    def test_forward_shapes_and_a_fit(self) -> None:
        rng = np.random.default_rng(2)
        left = rng.normal(loc=[-1.5, 0.0], scale=0.35, size=(30, 2))
        right = rng.normal(loc=[1.5, 0.0], scale=0.35, size=(30, 2))
        features = np.vstack([left, right])
        labels = np.array([0] * 30 + [1] * 30)
        model = MLP(
            hidden_units=8,
            task="classification",
            optimizer="adam",
            learning_rate=0.05,
            max_iter=60,
            random_state=0,
        ).fit(features, labels)
        scores = model.decision_function(features)
        probabilities = model.predict_proba(features)
        self.assertEqual(scores.shape, (60, 2))
        self.assertEqual(probabilities.shape, (60, 2))
        self.assertEqual(model.weights_input_.shape, (2, 8))
        self.assertEqual(model.weights_output_.shape, (8, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertGreater(model.score(features, labels), 0.9)
        self.assertGreater(model.loss_history_[0], model.loss_history_[-1])

        line_x = np.linspace(-1.0, 1.0, 25).reshape(-1, 1)
        line_y = 1.5 * line_x.ravel() + 0.25
        regressor = MLP(
            hidden_units=8,
            task="regression",
            optimizer="adam",
            learning_rate=0.05,
            max_iter=200,
            random_state=0,
        ).fit(line_x, line_y)
        self.assertEqual(regressor.predict(line_x).shape, (25,))
        self.assertEqual(regressor.decision_function(line_x).shape, (25, 1))
        self.assertGreater(regressor.score(line_x, line_y), 0.9)


def _flat_parameters(model: MLP) -> np.ndarray:
    assert model.weights_input_ is not None
    assert model.bias_hidden_ is not None
    assert model.weights_output_ is not None
    assert model.bias_output_ is not None
    return np.concatenate(
        [
            model.weights_input_.ravel(),
            model.bias_hidden_.ravel(),
            model.weights_output_.ravel(),
            model.bias_output_.ravel(),
        ]
    )


def _flat_gradient(model: MLP, features: np.ndarray, observed: np.ndarray) -> np.ndarray:
    gradients = model._gradients(features, observed, features.shape[0])
    return np.concatenate(
        [
            gradients["input"].ravel(),
            gradients["hidden_bias"].ravel(),
            gradients["output"].ravel(),
            gradients["output_bias"].ravel(),
        ]
    )


def _objective_at(model: MLP, features: np.ndarray, observed: np.ndarray, flat: np.ndarray) -> float:
    assert model.weights_input_ is not None
    assert model.bias_hidden_ is not None
    assert model.weights_output_ is not None
    assert model.bias_output_ is not None
    sizes = [
        model.weights_input_.size,
        model.bias_hidden_.size,
        model.weights_output_.size,
    ]
    pieces = np.split(flat, np.cumsum(sizes))
    model.weights_input_ = pieces[0].reshape(model.weights_input_.shape)
    model.bias_hidden_ = pieces[1].reshape(model.bias_hidden_.shape)
    model.weights_output_ = pieces[2].reshape(model.weights_output_.shape)
    model.bias_output_ = pieces[3].reshape(model.bias_output_.shape)
    return model._objective(features, observed)


if __name__ == "__main__":
    unittest.main()
