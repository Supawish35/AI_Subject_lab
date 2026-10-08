import unittest
import numpy as np
from main import calculate_univariate_divergence, calculate_multivariate_divergence

class TestDivergenceFormulas(unittest.TestCase):
    def test_identical_distributions(self):
        """Divergence between identical distributions must be zero."""
        u1, s1_sq = 2.0, 1.5
        u2, s2_sq = 2.0, 1.5
        d_uni = calculate_univariate_divergence(u1, s1_sq, u2, s2_sq)
        self.assertAlmostEqual(d_uni, 0.0, places=6)

        m1 = np.array([2.0, 3.0])
        cov1 = np.array([[1.5, 0.2], [0.2, 2.0]])
        d_multi = calculate_multivariate_divergence(m1, cov1, m1, cov1)
        self.assertAlmostEqual(d_multi, 0.0, places=4)

    def test_univariate_equals_multivariate_1d(self):
        """Multivariate formula with 1 dimension must exactly equal univariate formula."""
        u1, s1_sq = 3.5, 2.1
        u2, s2_sq = 1.2, 0.8

        d_uni = calculate_univariate_divergence(u1, s1_sq, u2, s2_sq)

        m1 = np.array([u1])
        cov1 = np.array([[s1_sq]])
        m2 = np.array([u2])
        cov2 = np.array([[s2_sq]])

        d_multi = calculate_multivariate_divergence(m1, cov1, m2, cov2)
        self.assertAlmostEqual(d_uni, d_multi, places=5)

    def test_symmetry(self):
        """d_ij must equal d_ji (symmetric divergence)."""
        u1, s1_sq = 1.0, 0.5
        u2, s2_sq = 4.0, 2.5
        d_12 = calculate_univariate_divergence(u1, s1_sq, u2, s2_sq)
        d_21 = calculate_univariate_divergence(u2, s2_sq, u1, s1_sq)
        self.assertAlmostEqual(d_12, d_21, places=6)

if __name__ == '__main__':
    unittest.main()
