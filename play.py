from __future__ import print_function

import pprint

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from network import BCPNN, NetworkManager, BCPNNFast, Protocol
from plotting_functions import plot_state_variables_vs_time, plot_network_activity, plot_network_activity_angle
from plotting_functions import  plot_adaptation_dynamics, plot_weight_matrix, plot_winning_pattern, plot_sequence
from analysis_functions import calculate_recall_success
# np.set_printoptions(suppress=True, precision=2)

# Patterns parameters
hypercolumns = 4
minicolumns = 20

# Manager properties
dt = 0.001
T_recall = 5.0
values_to_save = ['o']


# Protocol
training_time = 0.1
inter_sequence_interval = 2.0
inter_pulse_interval = 0.0
epochs = 1

# Build the network
nn = BCPNNFast(hypercolumns, minicolumns)
nn.k_inner = True

# Build the manager
manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

# Build the protocol for
patterns = [0, 1, 2, 3, 4]
protocol = Protocol()
protocol.simple_protocol(patterns_indexes=patterns, training_time=training_time,
                         inter_pulse_interval=inter_pulse_interval, inter_sequence_interval=inter_sequence_interval,
                         epochs=epochs)


# Train
manager.run_network_protocol(protocol, verbose=False, values_to_save_epoch=None, reset=True, empty_history=True)

manager.run_network_recall(T_recall=T_recall, T_cue=0.1, I_cue=0, reset=True, empty_history=True)


success = calculate_recall_success(manager, T_recall=T_recall, I_cue=0, T_cue=0.100, n=5, patterns_indexes=patterns)
print(success)






