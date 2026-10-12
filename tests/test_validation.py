"""Metrics, standardization, and K-fold partitions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ml_from_scratch.linear_models import LogisticRegression
from ml_from_scratch.utils import (
    StandardScaler,
    accuracy_score,
    cross_val_score,
    k_fold_split,
    r2_score,
    roc_auc_score,
    train_test_split,
)
from ml_from_scratch.utils.metrics import precision_score


class MetricBaselineTests(unittest.TestCase):
    def test_accuracy_precision_and_r2(self) -> None:
        observed = np.array([0, 0, 0, 1])
        predicted = np.array([0, 0, 0, 0])
        self.assertAlmostEqual(accuracy_score(observed, predicted), 0.75)
        self.assertAlmostEqual(precision_score(observed, predicted, pos_label=0), 0.75)
        perfect = np.array([1.0, 2.0, 3.0])
        self.assertEqual(r2_score(perfect, perfect), 1.0)
        self.assertAlmostEqual(r2_score(perfect, np.full(3, 2.0)), 0.0)

    def test_roc_auc_of_a_known_ranking(self) -> None:
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.35, 0.8])
        self.assertAlmostEqual(roc_auc_score(labels, scores), 0.75)
        self.assertAlmostEqual(roc_auc_score(labels, np.zeros(4)), 0.5)


class SplitTests(unittest.TestCase):
    def test_standardizer_and_holdout(self) -> None:
        rng = np.random.default_rng(0)
        features = rng.normal(loc=3.0, scale=2.0, size=(20, 2))
        transformed = StandardScaler().fit_transform(features)
        np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=1e-10)
        labels = np.arange(20)
        train_x, test_x, train_y, test_y = train_test_split(
            features,
            labels,
            test_size=0.25,
            shuffle=False,
        )
        self.assertEqual(test_x.shape[0], 5)
        self.assertEqual(train_x.shape[0], 15)
        np.testing.assert_array_equal(test_y, labels[-5:])
        np.testing.assert_array_equal(train_y, labels[:-5])

    def test_k_fold_indices_are_a_partition(self) -> None:
        n_samples = 23
        n_splits = 5
        covered = np.zeros(n_samples, dtype=int)
        for train_idx, test_idx in k_fold_split(
            n_samples,
            n_splits=n_splits,
            shuffle=True,
            random_state=1,
        ):
            self.assertEqual(np.intersect1d(train_idx, test_idx).size, 0)
            self.assertEqual(train_idx.size + test_idx.size, n_samples)
            self.assertEqual(np.unique(test_idx).size, test_idx.size)
            covered[test_idx] += 1
        np.testing.assert_array_equal(covered, np.ones(n_samples))

    def test_unshuffled_folds_are_contiguous_blocks(self) -> None:
        folds = list(k_fold_split(10, n_splits=5, shuffle=False))
        self.assertEqual(len(folds), 5)
        np.testing.assert_array_equal(folds[0][1], np.array([0, 1]))
        np.testing.assert_array_equal(folds[-1][1], np.array([8, 9]))

    def test_stratified_folds_preserve_class_balance(self) -> None:
        labels = np.array([0] * 20 + [1] * 20)
        for _train_idx, test_idx in k_fold_split(
            40,
            n_splits=4,
            shuffle=True,
            random_state=0,
            y=labels,
        ):
            self.assertEqual(int(np.sum(labels[test_idx] == 0)), 5)
            self.assertEqual(int(np.sum(labels[test_idx] == 1)), 5)

    def test_cross_validated_scores_have_one_entry_per_fold(self) -> None:
        features = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
        labels = (features.ravel() > 0.0).astype(int)
        scores = cross_val_score(
            LogisticRegression(method="newton", max_iter=30, tol=1e-8),
            features,
            labels,
            n_splits=4,
            stratify=True,
            random_state=0,
        )
        self.assertEqual(scores.shape, (4,))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertGreater(float(scores.mean()), 0.8)


if __name__ == "__main__":
    unittest.main()
