"""Mathematical invariants for p > n Gaussian factor projection."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import FactorAnalysis
from threadpoolctl import threadpool_limits

from scripts.latent_space_benchmark import (
    fit_model, observed_log_likelihood, posterior_scores,
    raw_observed_log_likelihood, transform_new_objects, varimax_rotation,
)


class LatentSpaceMathTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261014)
        self.loadings = rng.normal(size=(40, 3))
        self.noise = rng.uniform(.2, .8, size=40)
        self.x = rng.normal(size=(16, 3)) @ self.loadings.T
        self.x += rng.normal(size=self.x.shape) * np.sqrt(self.noise)

    def test_p_greater_than_n_and_duplicate_features_remain_finite(self):
        raw = np.column_stack([self.x, self.x[:, 0], self.x[:, 0]])
        with threadpool_limits(limits=1):
            parameters, diagnostics = fit_model(raw, 3)
        self.assertGreater(raw.shape[1], raw.shape[0])
        self.assertTrue(np.isfinite(transform_new_objects(raw, parameters)).all())
        self.assertGreaterEqual(parameters["noise_variance"].min(), 1e-6)
        self.assertTrue(np.isfinite(diagnostics["posterior_system_condition"]))

    def test_posterior_and_log_density_match_sklearn_after_floor(self):
        with threadpool_limits(limits=1):
            model = FactorAnalysis(n_components=3, svd_method="lapack", max_iter=2000).fit(self.x)
        model.noise_variance_ = np.maximum(model.noise_variance_, 1e-6)
        calculated = posterior_scores(self.x, model.components_.T,
                                      model.noise_variance_, model.mean_)
        np.testing.assert_allclose(calculated, model.transform(self.x), atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(observed_log_likelihood(self.x, model.components_.T,
                                                          model.noise_variance_, model.mean_),
                                   model.score_samples(self.x), atol=1e-8, rtol=1e-8)

    def test_missing_coordinates_are_marginalized_and_zero_is_observed(self):
        x = self.x[:2].copy()
        x[0, :10] = np.nan
        x[1] = np.nan
        projected = posterior_scores(x, self.loadings, self.noise)
        expected = posterior_scores(x[:1, 10:], self.loadings[10:], self.noise[10:])
        np.testing.assert_allclose(projected[:1], expected)
        np.testing.assert_allclose(projected[1], np.zeros(3))
        ll = observed_log_likelihood(x, self.loadings, self.noise, np.zeros(40))
        self.assertEqual(ll[1], 0.)
        self.assertTrue(np.isfinite(ll).all())
        x[1, 0] = 0.
        self.assertLess(observed_log_likelihood(x[1:], self.loadings, self.noise, np.zeros(40))[0], 0.)

    def test_varimax_preserves_covariance_reconstruction_and_distances(self):
        rotation = varimax_rotation(self.loadings)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        scores = posterior_scores(self.x, self.loadings, self.noise)
        rotated_l, rotated_s = self.loadings @ rotation, scores @ rotation
        np.testing.assert_allclose(rotated_l @ rotated_l.T,
                                   self.loadings @ self.loadings.T, atol=1e-12)
        np.testing.assert_allclose(rotated_s @ rotated_l.T,
                                   scores @ self.loadings.T, atol=1e-12)
        np.testing.assert_allclose(pdist(rotated_s), pdist(scores), atol=1e-12)
        np.testing.assert_allclose(posterior_scores(self.x, rotated_l, self.noise),
                                   rotated_s, atol=1e-12)

    def test_serialized_parameters_transform_without_refitting(self):
        with threadpool_limits(limits=1):
            parameters, _ = fit_model(self.x, 3)
        parameters["trait_order"] = [f"trait{i}" for i in range(self.x.shape[1])]
        new = self.x[:2] * 1.1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parameters.joblib"
            joblib.dump(parameters, path)
            loaded = joblib.load(path)
        np.testing.assert_array_equal(transform_new_objects(new, loaded),
                                      transform_new_objects(new, parameters))
        # Raw-scale likelihood includes the Jacobian for the fixed scaler.
        standardized = (new - loaded["training_mean"]) / loaded["training_scale"]
        expected = observed_log_likelihood(standardized, loaded["loadings"],
                                           loaded["noise_variance"], loaded["standardized_model_mean"])
        expected -= np.log(loaded["training_scale"]).sum()
        np.testing.assert_allclose(raw_observed_log_likelihood(new, loaded), expected)

    def test_nonpositive_noise_and_invalid_new_feature_shape_are_rejected(self):
        noise = self.noise.copy()
        noise[0] = 0.
        with self.assertRaises(ValueError):
            posterior_scores(self.x, self.loadings, noise)
        with threadpool_limits(limits=1):
            parameters, _ = fit_model(self.x, 3)
        with self.assertRaises(ValueError):
            transform_new_objects(self.x[:, :-1], parameters)


if __name__ == "__main__":
    unittest.main()
