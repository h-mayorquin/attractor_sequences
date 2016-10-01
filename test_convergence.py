import numpy as np
import numpy.testing as npt
import unittest

from connectivity_functions import get_beta, get_w
from connectivity_functions import calculate_probability, calculate_coactivations
from data_transformer import build_ortogonal_patterns
from convergence_functions import test_convergence_ratios
from network import BCPNN


class TestConvergence(unittest.TestCase):
    """
    This should test all the convergence properties of the network under
    the conditions of the free recall paper (parameter values)
    """

    def test_basic_convergence(self):
        """
        This will test convergence for the most basic
        configuration. Only three parameters and no-adaptation
        """

        hypercolumns = 10
        minicolumns = 10

        patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
        patterns = list(patterns_dic.values())

        P = calculate_coactivations(patterns)
        p = calculate_probability(patterns)

        w = get_w(P, p)
        beta = get_beta(p)

        dt = 0.01
        T = 1
        time = np.arange(0, T + dt, dt)

        g_a = 0  # No adaptation
        g_beta = 1.0  # No bias gain
        g_w = 1.0  # No weight gain
        prng = np.random.RandomState(seed=0)

        nn = BCPNN(hypercolumns, minicolumns, beta, w, p_pre=p, p_post=p, p_co=P,
                   g_a=g_a, g_beta=g_beta, g_w=g_w, prng=prng)
        N = 50

        fraction_of_convergence, fraction_of_well_behaved = test_convergence_ratios(nn, N, time, patterns)

        self.assertAlmostEqual(first=fraction_of_convergence, second=1.0)
        self.assertAlmostEqual(first=fraction_of_well_behaved, second=1.0)


if __name__ == '__main__':
    unittest.main()