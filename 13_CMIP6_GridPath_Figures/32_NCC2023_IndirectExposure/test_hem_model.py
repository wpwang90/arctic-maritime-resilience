"""Synthetic fixtures are unit tests only, never empirical research inputs."""
import unittest

import numpy as np
from numpy.testing import assert_allclose

from hem_model import MRIO, joint_extraction, overlap_safe_share


class TestHEM(unittest.TestCase):
    def setUp(self):
        self.z = np.array([[10., 20., 5.], [8., 12., 16.], [7., 10., 14.]])
        self.y = np.array([[40., 20.], [20., 35.], [30., 20.]])
        self.x = self.z.sum(1) + self.y.sum(1)
        self.v = self.x - self.z.sum(0)
        self.mrio = MRIO(self.z, self.y, self.x, self.v)

    def test_zero_shock(self):
        result = joint_extraction(self.mrio, np.zeros_like(self.z))
        assert_allclose(result['forward_output_exposure'], 0)
        assert_allclose(result['backward_intermediate_exposure'], 0)

    def test_published_forward_equation(self):
        q = np.zeros_like(self.z); q[0, 1] = .6
        result = joint_extraction(self.mrio, q)
        b = self.z * (1-q) / self.x[:, None]
        exact = self.x - self.v @ np.linalg.inv(np.eye(3) - b)
        assert_allclose(result['forward_output_exposure'], exact, atol=1e-12)

    def test_published_backward_equation(self):
        q = np.zeros_like(self.z); q[0, 1] = .6
        result = joint_extraction(self.mrio, q)
        a = self.z * (1-q) / self.x[None, :]
        exact = self.x - np.linalg.solve(np.eye(3)-a, self.y.sum(1))
        assert_allclose(result['backward_intermediate_exposure'], exact, atol=1e-12)

    def test_first_hit_decomposition(self):
        q = np.full_like(self.z, .25)
        result = joint_extraction(self.mrio, q)
        assert_allclose(result['forward_first_hit'] + result['forward_higher_order_exposure'],
                        result['forward_output_exposure'])
        self.assertTrue((result['forward_higher_order_exposure'] >= 0).all())

    def test_complete_intermediate_extraction(self):
        r = joint_extraction(self.mrio, np.ones_like(self.z))
        assert_allclose(r['forward_remaining_output'], self.v)
        assert_allclose(r['forward_higher_order_exposure'], 0, atol=1e-12)

    def test_nonlinearity_not_sum(self):
        q1 = np.zeros_like(self.z); q1[0, 1] = 1
        q2 = np.zeros_like(self.z); q2[1, 2] = 1
        one = joint_extraction(self.mrio, q1)['forward_output_exposure']
        two = joint_extraction(self.mrio, q2)['forward_output_exposure']
        both = joint_extraction(self.mrio, q1+q2)['forward_output_exposure']
        self.assertFalse(np.allclose(one+two, both))

    def test_reconnection_reduces_total_exposure(self):
        wo = joint_extraction(self.mrio, np.full_like(self.z, .5))
        wi = joint_extraction(self.mrio, np.full_like(self.z, .2))
        self.assertTrue((wi['forward_output_exposure'] <= wo['forward_output_exposure']).all())

    def test_overlap_is_not_double_counted(self):
        hits = [[True, True], [False, False], [False, True]]
        self.assertAlmostEqual(overlap_safe_share([40, 30, 30], hits), .7)

    def test_invalid_share_rejected(self):
        with self.assertRaises(ValueError):
            joint_extraction(self.mrio, np.full_like(self.z, 1.01))

    def test_unbalanced_table_rejected(self):
        with self.assertRaises(ValueError):
            MRIO(self.z, self.y, self.x, self.v + 1).validate()

    def test_final_consumption_kept_separate(self):
        qy = np.zeros_like(self.y); qy[0, 1] = .5
        r = joint_extraction(self.mrio, np.zeros_like(self.z), qy)
        assert_allclose(r['forward_output_exposure'], 0)
        self.assertEqual(r['final_consumption_exposure_by_supplier_buyer'].sum(), 10.)
        self.assertGreater(r['backward_final_demand_exposure'].sum(), 10.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
