"""Observed-mask mathematics and honest validation for missing-data models."""
import unittest
import numpy as np
from threadpoolctl import threadpool_limits

from src.corpus_pipeline.incomplete_latent import (
    Priors, chain_diagnostics, gaussian_precision_sample, gibbs_factor_analysis,
    compare_geometry, holdout_mask, prediction_mean, ridge_factorization, validation_folds,
)


class IncompleteLatentTests(unittest.TestCase):
    def test_precision_sampler_has_correct_non_diagonal_covariance(self):
        rng = np.random.default_rng(22)
        precision = np.array([[3., 1.2], [1.2, 1.]])
        h = np.array([1., -1.])
        draws = gaussian_precision_sample(np.broadcast_to(precision, (50000, 2, 2)),
                                          np.broadcast_to(h, (50000, 2)), rng)
        np.testing.assert_allclose(draws.mean(axis=0), np.linalg.solve(precision, h), atol=.02)
        np.testing.assert_allclose(np.cov(draws.T), np.linalg.inv(precision), atol=.035)

    def test_ridge_objective_decreases_and_missing_is_not_zero(self):
        x = np.array([[0., 1., np.nan], [2., np.nan, 3.], [np.nan, 4., 5.], [1., 3., 2.]])
        fit = ridge_factorization(x, 1, .3)
        self.assertTrue(np.all(np.diff(fit['history']) <= 1e-9))
        changed = ridge_factorization(np.nan_to_num(x), 1, .3)
        self.assertGreater(np.linalg.norm(fit['prediction'] - changed['prediction']), .2)

    def test_empty_rows_and_columns_do_not_add_pseudo_observations(self):
        x = np.array([[1., 2., np.nan], [3., 4., np.nan], [np.nan] * 3])
        fit = ridge_factorization(x, 1, 1.)
        np.testing.assert_allclose(fit['scores'][-1], 0., atol=1e-12)
        np.testing.assert_allclose(fit['loadings'][-1], 0., atol=1e-12)
        self.assertAlmostEqual(fit['mean'][-1], 2.5)

    def test_validation_only_hides_exact_observations_and_preserves_overlap(self):
        x = np.arange(30, dtype=float).reshape(6, 5)
        x[0, 1:] = np.nan
        exact = np.isfinite(x)
        exact[:, -1] = False
        target = holdout_mask(x, exact, 21, fraction=.8)
        self.assertFalse((target & ~exact).any())
        train = np.isfinite(x) & ~target
        self.assertTrue((train.sum(axis=1) >= 1).all())
        self.assertTrue((train.sum(axis=0) >= 2).all())
        self.assertFalse(target[0, 0])

    def test_rank_diagnostics_distinguish_independent_from_shifted_chains(self):
        rng = np.random.default_rng(34)
        x = rng.normal(size=(4, 2000, 3))
        good = chain_diagnostics(x)
        self.assertLess(good['max_rank_split_rhat'], 1.01)
        self.assertGreater(good['min_bulk_ess'], 4000)
        x[0] += 3
        self.assertGreater(chain_diagnostics(x)['max_rank_split_rhat'], 1.1)

    def test_disjoint_folds_cover_extreme_and_ordinary_eligible_cells_once(self):
        x = np.full((12, 6), 3.)
        x[0, 0] = 0.
        exact = np.ones(x.shape, bool)
        targets, coverage = validation_folds(x, exact, 8100)
        np.testing.assert_array_equal(np.stack(targets).sum(0), 1)
        self.assertEqual(coverage['tested_once'], x.size)
        self.assertEqual(coverage['eligible_not_allocated'], 0)
        for target in targets:
            self.assertTrue(((~target).sum(1) >= 1).all())
            self.assertTrue(((~target).sum(0) >= 2).all())

    def test_collapsed_solution_has_no_interpretable_geometry(self):
        model = {'loadings': np.ones((3, 1)), 'scores': np.ones((5, 1)),
                 'effective_rank_threshold_0_01': 0}
        result = compare_geometry(model, model)
        self.assertIsNone(result['distance_spearman'])
        self.assertIsNone(result['worst_axis_congruence'])

    def test_gibbs_prediction_is_invariant_to_joint_orthogonal_rotation(self):
        x = np.array([[0., 1., np.nan], [2., np.nan, 3.], [np.nan, 4., 5.], [1., 3., 2.]])
        draws = gibbs_factor_analysis(x, 2, warmup=30, draws=60, chains=2, seed=19)
        original = prediction_mean(draws)
        rot = np.array([[.6, -.8], [.8, .6]])
        draws['scores'] = draws['scores'] @ rot
        draws['loadings'] = draws['loadings'] @ rot
        np.testing.assert_allclose(prediction_mean(draws), original, atol=1e-12)

    def test_gibbs_missing_entry_never_creates_a_zero_observation(self):
        # Almost no shared signal; posterior intercept should reflect high
        # observed scores, rather than 75% working zeros in the computation.
        x = np.full((12, 4), np.nan)
        for i in range(12):
            x[i, i % 4] = 4.
        draws = gibbs_factor_analysis(x, 1, warmup=100, draws=500, chains=2, seed=7,
                                     priors=Priors(loading_sd=.1))
        self.assertTrue((draws['mean'].mean(axis=(0, 1)) > 3.5).all())

    def test_synthetic_low_rank_signal_predicts_hidden_entries(self):
        rng = np.random.default_rng(31)
        u = np.linspace(-2, 2, 24)[:, None]
        load = np.array([[-1., -.7, .5, .8, 1., -.9]])
        truth = 2.5 + u @ load
        hidden = rng.random(truth.shape) < .3
        x = np.where(hidden, np.nan, truth + rng.normal(0, .08, truth.shape))
        with threadpool_limits(limits=1):
            fit = ridge_factorization(x, 1, .1)
        error = np.sqrt(np.mean((fit['prediction'][hidden] - truth[hidden]) ** 2))
        baseline = np.broadcast_to(np.nanmean(x, axis=0), x.shape)
        baseline_error = np.sqrt(np.mean((baseline[hidden] - truth[hidden]) ** 2))
        self.assertLess(error, baseline_error / 4)

    def test_bayesian_model_recovers_a_real_signal_without_filling_gaps(self):
        rng = np.random.default_rng(31)
        truth = 2.5 + np.linspace(-2, 2, 24)[:, None] @ np.array([[-1., -.7, .5, .8, 1., -.9]])
        hidden = rng.random(truth.shape) < .3
        x = np.where(hidden, np.nan, truth + rng.normal(0, .08, truth.shape))
        with threadpool_limits(limits=1):
            draws = gibbs_factor_analysis(x, 1, warmup=300, draws=600, chains=2, seed=31)
        prediction = prediction_mean(draws)
        error = np.sqrt(np.mean((prediction[hidden] - truth[hidden]) ** 2))
        baseline = np.broadcast_to(np.nanmean(x, axis=0), x.shape)
        baseline_error = np.sqrt(np.mean((baseline[hidden] - truth[hidden]) ** 2))
        self.assertLess(error, baseline_error / 3)


if __name__ == '__main__':
    unittest.main()
