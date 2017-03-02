import numpy as np
import numpy.testing as npt
import unittest

from analysis_functions import calculate_recall_success
from network import BCPNNFast, NetworkManager, Protocol


class TestConvergence(unittest.TestCase):
    """
    This should test all the convergence properties of the network under
    the conditions of the free recall paper (parameter values)
    """

    def test_basic_convergence(self):
        """
        This tests a simple network for recall percentage.
        """

        # Patterns parameters
        hypercolumns = 4
        minicolumns = 20

        # Manager properties
        dt = 0.001
        T_recall = 4.0
        values_to_save = ['o']

        # Protocol
        training_time = 0.1
        inter_sequence_interval = 2.0
        inter_pulse_interval = 0.0
        epochs = 1

        # Build the network
        nn = BCPNNFast(hypercolumns, minicolumns)
        nn.k_inner = False

        # Build the manager
        manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

        # Build the protocol for
        patterns = [0, 1, 2, 3, 4]
        protocol = Protocol()
        protocol.simple_protocol(patterns_indexes=patterns, training_time=training_time,
                                 inter_pulse_interval=inter_pulse_interval,
                                 inter_sequence_interval=inter_sequence_interval,
                                 epochs=epochs)

        # Train
        manager.run_network_protocol(protocol, verbose=False, values_to_save_epoch=None, reset=True, empty_history=True)

        manager.run_network_recall(T_recall=T_recall, T_cue=0.1, I_cue=0, reset=True, empty_history=True)

        success = calculate_recall_success(manager, T_recall=T_recall, I_cue=0, T_cue=0.100, n=25,
                                           patterns_indexes=patterns)

        self.assertAlmostEqual(success, 100.0)

if __name__ == '__main__':
    unittest.main()