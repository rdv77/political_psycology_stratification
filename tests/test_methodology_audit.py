"""Mathematical invariants needed to interpret the audit correctly."""
import unittest

import numpy as np

from scripts.methodology_audit import aligned_phi, congruence, leading_retained


class AuditMathTests(unittest.TestCase):
    def test_factor_matching_is_invariant_to_order_and_sign(self):
        loadings = np.array([[.8, .1], [.7, -.2], [.1, .9], [-.2, .8]])
        equivalent = loadings[:, [1, 0]] * np.array([-1., 1.])
        np.testing.assert_allclose(congruence(loadings, equivalent), [1., 1.])

    def test_structure_recovers_oblique_correlations_after_reordering(self):
        loadings = np.array([[.8, .1, .2], [.7, -.2, .1], [.1, .9, .2], [-.2, .8, .7]])
        phi = np.array([[1., .3, -.2], [.3, 1., .1], [-.2, .1, 1.]])
        structure = loadings @ phi
        order = [2, 0, 1]
        sorted_loadings = loadings[:, order]
        sorted_structure = structure[:, order]
        recovered = aligned_phi(sorted_loadings, sorted_structure)
        np.testing.assert_allclose(recovered, phi[np.ix_(order, order)], atol=1e-12)
        np.testing.assert_allclose(sorted_loadings @ recovered @ sorted_loadings.T,
                                   loadings @ phi @ loadings.T, atol=1e-12)

    def test_parallel_retention_stops_at_first_failure(self):
        self.assertEqual(leading_retained(np.array([8., 3., 1., 1.5]),
                                          np.array([4., 2., 1.2, 1.1])), 2)

    def test_singular_score_projection_has_a_bounded_solution(self):
        z = np.array([[-1., -1.], [0., 0.], [1., 1.]])
        correlation = np.ones((2, 2))
        loadings = np.ones((2, 1))
        stable = z @ np.linalg.pinv(correlation, rcond=1e-10) @ loadings
        np.testing.assert_allclose(stable[:, 0], [-1., 0., 1.])


if __name__ == "__main__":
    unittest.main()
