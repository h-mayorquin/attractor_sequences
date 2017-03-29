from __future__ import print_function

import pprint

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from network import BCPNN, NetworkManager, BCPNNFast, Protocol
from plotting_functions import plot_state_variables_vs_time, plot_network_activity, plot_network_activity_angle
from plotting_functions import plot_adaptation_dynamics, plot_weight_matrix
from analysis_functions import calculate_compression_factor, calculate_recall_success, calculate_timings
from plotting_functions import plot_winning_pattern, plot_sequence, plot_network_activity_angle

# np.set_printoptions(suppress=True, precision=2)

# Patterns parameters
hypercolumns = 4
minicolumns = 30

# Manager properties
dt = 0.001
T_recall = 3.0
values_to_save = ['o']

compression_list = []


# Protocol
training_time = 0.1
inter_sequence_interval = 1.0
inter_pulse_interval = 0.0
epochs = 3

# Build the network
nn = BCPNNFast(hypercolumns, minicolumns)
nn.k_inner = False

# Build the manager
manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

# Build the protocol for
protocol = Protocol()

number_of_sequences = 3
half_width = 2
units_to_overload = [10]
chain = []

protocol = Protocol()
chain = protocol.create_overload_chain(number_of_sequences, half_width, units_to_overload)
protocol.cross_protocol(chain, training_time, inter_sequence_interval, epochs)
# Train
manager.run_network_protocol(protocol, verbose=False, values_to_save_epoch=None, reset=True, empty_history=True)

# Recall
if True:
    manager.run_network_recall(T_recall=T_recall, T_cue=0.1, I_cue=0, reset=True, empty_history=True)
    manager.run_network_recall(T_recall=T_recall, T_cue=0.1, I_cue=5, reset=True, empty_history=False)
    manager.run_network_recall(T_recall=T_recall, T_cue=0.1, I_cue=11, reset=True, empty_history=False)
    manager.run_network_recall(T_recall=T_recall, T_cue=0.1, I_cue=15, reset=True, empty_history=False)

    # Timings
    timings = calculate_timings(manager, remove=0.010)
    print(timings)

    plot_winning_pattern(manager, remove=0.010)
    plt.show()

if True:
    n = 5
    for sequence in chain:
        success = calculate_recall_success(manager, T_recall=T_recall, I_cue=sequence[0], T_cue=0.1, n=n,
                                        patterns_indexes=sequence)
        print(success)
