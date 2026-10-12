"""K-means, PCA, mixtures, and generative classifiers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ml_from_scratch.probabilistic import (
    BernoulliNaiveBayes,
    GaussianDiscriminantAnalysis,
    GaussianMixture,
    GaussianNaiveBayes,
    MultinomialNaiveBayes,
)
from ml_from_scratch.unsupervised import PCA, KMeans, pairwise_squared_distances


class KMeansTests(unittest.TestCase):
    def test_squared_distance_matches_the_definition(self) -> None:
        rng = np.random.default_rng(0)
        left = rng.normal(size=(8, 3))
        right = rng.normal(size=(4, 3))
        expanded = pairwise_squared_distances(left, right)
        direct = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
        self.assertEqual(expanded.shape, (8, 4))
        np.testing.assert_allclose(expanded, direct)

    def test_lloyd_separates_two_blobs(self) -> None:
        rng = np.random.default_rng(1)
        left = rng.normal(loc=[-3.0, 0.0], scale=0.3, size=(30, 2))
        right = rng.normal(loc=[3.0, 0.0], scale=0.3, size=(30, 2))
        features = np.vstack([left, right])
        model = KMeans(n_clusters=2, n_init=4, random_state=0).fit(features)
        assert model.labels_ is not None and model.cluster_centers_ is not None
        self.assertEqual(model.cluster_centers_.shape, (2, 2))
        self.assertEqual(model.labels_.shape, (60,))
        self.assertEqual(len(np.unique(model.labels_[:30])), 1)
        self.assertNotEqual(model.labels_[0], model.labels_[-1])
        distances = pairwise_squared_distances(features, model.cluster_centers_)
        inertia = float(np.sum(distances[np.arange(60), model.labels_]))
        self.assertAlmostEqual(inertia, float(model.inertia_), places=8)


class PCATests(unittest.TestCase):
    def test_svd_and_eigen_agree_and_shapes_match(self) -> None:
        rng = np.random.default_rng(2)
        direction = np.array([1.0, -2.0, 0.5])
        direction /= np.linalg.norm(direction)
        scores = rng.normal(size=40)
        features = scores[:, None] * direction + 0.01 * rng.normal(size=(40, 3))
        by_svd = PCA(n_components=2, method="svd").fit(features)
        by_eigen = PCA(n_components=2, method="eigen").fit(features)
        assert by_svd.components_ is not None and by_eigen.components_ is not None
        assert by_svd.explained_variance_ratio_ is not None
        self.assertEqual(by_svd.components_.shape, (2, 3))
        projected = by_svd.transform(features)
        self.assertEqual(projected.shape, (40, 2))
        np.testing.assert_allclose(by_svd.components_, by_eigen.components_, atol=1e-8)
        np.testing.assert_allclose(
            by_svd.explained_variance_,
            by_eigen.explained_variance_,
            atol=1e-8,
        )
        self.assertGreater(float(by_svd.explained_variance_ratio_[0]), 0.95)
        self.assertLessEqual(float(np.sum(by_svd.explained_variance_ratio_)), 1.0 + 1e-8)


class GenerativeModelTests(unittest.TestCase):
    def test_gmm_responsibilities_and_separation(self) -> None:
        rng = np.random.default_rng(3)
        left = rng.normal(loc=[-4.0, 0.0], scale=0.35, size=(40, 2))
        right = rng.normal(loc=[4.0, 0.5], scale=0.35, size=(40, 2))
        features = np.vstack([left, right])
        model = GaussianMixture(n_components=2, n_init=2, random_state=0, max_iter=40).fit(features)
        responsibilities = model.predict_proba(features)
        self.assertEqual(responsibilities.shape, (80, 2))
        np.testing.assert_allclose(responsibilities.sum(axis=1), 1.0)
        labels = model.predict(features)
        self.assertEqual(len(np.unique(labels[:40])), 1)
        self.assertNotEqual(labels[0], labels[-1])
        history = np.asarray(model.log_likelihood_history_)
        self.assertTrue(np.all(np.diff(history) >= -1e-6))

    def test_shared_gda_linear_score_matches_the_posterior(self) -> None:
        rng = np.random.default_rng(4)
        left = rng.normal(loc=[-2.5, 0.0], scale=0.4, size=(30, 2))
        right = rng.normal(loc=[2.5, 0.2], scale=0.4, size=(30, 2))
        features = np.vstack([left, right])
        labels = np.array([0] * 30 + [1] * 30)
        model = GaussianDiscriminantAnalysis(covariance="shared").fit(features, labels)
        self.assertEqual(model.score(features, labels), 1.0)
        assert model.linear_coef_ is not None and model.linear_intercept_ is not None
        scores = features @ model.linear_coef_.T + model.linear_intercept_
        scores = scores - scores.max(axis=1, keepdims=True)
        bayes = np.exp(scores)
        bayes /= bayes.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(model.predict_proba(features), bayes, atol=1e-8)
        self.assertEqual(model.predict_proba(features).shape, (60, 2))

    def test_naive_bayes_baselines(self) -> None:
        rng = np.random.default_rng(5)
        left = rng.normal(loc=[-2.0, 0.0], scale=0.3, size=(25, 2))
        right = rng.normal(loc=[2.0, 0.0], scale=0.3, size=(25, 2))
        features = np.vstack([left, right])
        labels = np.array([0] * 25 + [1] * 25)
        gaussian = GaussianNaiveBayes().fit(features, labels)
        self.assertEqual(gaussian.score(features, labels), 1.0)
        np.testing.assert_allclose(gaussian.predict_proba(features).sum(axis=1), 1.0)

        bits = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        bit_labels = np.array([0, 0, 1, 1])
        bernoulli = BernoulliNaiveBayes().fit(bits, bit_labels)
        self.assertEqual(bernoulli.score(bits, bit_labels), 1.0)

        counts = np.array([[4.0, 0.0], [3.0, 1.0], [0.0, 5.0], [1.0, 4.0]])
        count_labels = np.array([0, 0, 1, 1])
        multinomial = MultinomialNaiveBayes().fit(counts, count_labels)
        self.assertEqual(multinomial.predict_proba(counts).shape, (4, 2))
        self.assertEqual(multinomial.score(counts, count_labels), 1.0)


if __name__ == "__main__":
    unittest.main()
