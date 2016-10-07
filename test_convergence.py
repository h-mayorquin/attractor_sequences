import numpy as np
import numpy.testing as npt
import unittest

from connectivity_functions import get_beta, get_w
from connectivity_functions import calculate_probability, calculate_coactivations
from data_transformer import build_ortogonal_patterns
from convergence_functions import calculate_convergence_ratios
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

        # Run the network 50 times
        N = 50
        fraction_of_convergence, fraction_of_well_behaved = calculate_convergence_ratios(nn, N, time, patterns)

        self.assertAlmostEqual(first=fraction_of_convergence, second=1.0)
        self.assertAlmostEqual(first=fraction_of_well_behaved, second=1.0)

    def test_clamping_vs_bias_convergence_to_clamping(self):
        """
        This test that the BCPNN converges to the clamped value no matter the value
        of the biases of the adaptation, weights or bias.
        """

        hypercolumns = 4
        minicolumns = 4

        patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
        patterns = list(patterns_dic.values())

        P = calculate_coactivations(patterns)
        p = calculate_probability(patterns)

        w = get_w(P, p)
        beta = get_beta(p)

        dt = 0.01
        T_simulation = 1.0
        simulation_time = np.arange(0, T_simulation + dt, dt)

        prng = np.random.RandomState(seed=0)

        g_a_set = np.arange(0, 110, 10)
        g_beta_set = np.arange(0, 5, 0.5)
        g_w_set = np.arange(0, 3.2, 0.2)

        # Pattern to clamp
        g_I = 20.0
        I = np.array((1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))

        # Test the error
        for index_1, g_a in enumerate(g_a_set):
            for index_2, g_beta in enumerate(g_beta_set):
                for index_3, g_w in enumerate(g_w_set):
                    nn = BCPNN(hypercolumns, minicolumns, beta, w, p_pre=p, p_post=p, p_co=P,
                               g_a=g_a, g_beta=g_beta, g_w=g_w, g_I=g_I, prng=prng, k=0)

                    nn.randomize_pattern()

                    # This is the training
                    nn.run_network_simulation(simulation_time, I=I)
                    final = nn.o
                    point_error = np.linalg.norm(I - final)
                    self.assertAlmostEqual(point_error, 0)

if __name__ == '__main__':
    unittest.main()