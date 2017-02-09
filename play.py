from __future__ import print_function

import pprint

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from network import BCPNN, NetworkManager, BCPNNFast, Protocol
from data_transformer import build_ortogonal_patterns
from plotting_functions import plot_state_variables_vs_time, plot_network_activity, plot_network_activity_angle
from plotting_functions import  plot_adaptation_dynamics, plot_weight_matrix, plot_winning_pattern, plot_sequence
# np.set_printoptions(suppress=True, precision=2)

# Patterns parameters
hypercolumns = 4
minicolumns = 30
n_patterns = 5  # Number of patterns

# Manager properties
dt = 0.001
T_recalling = 10.0
values_to_save = ['o', 's', 'z_pre', 'z_post', 'p_pre', 'p_post', 'p_co', 'z_co', 'w',
                  'p', 'k_d', 'z_pre_ampa', 'z_post_ampa', 'p_pre_ampa', 'p_post_ampa', 'p_co_ampa', 'z_co_ampa', 'w_ampa']

# Protocol
training_time = 0.1
inter_sequence_interval = 2.0
inter_pulse_interval = 0.0
epochs = 3

# Build patterns
patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
patterns = list(patterns_dic.values())
patterns = patterns[:n_patterns]

# Build the network
nn = BCPNNFast(hypercolumns, minicolumns)

nn.k_inner = False


# Build the manager
manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

# Build the protocol for training
protocol = Protocol()
protocol.simple_protocol(patterns, training_time=training_time, inter_pulse_interval=inter_pulse_interval,
                         inter_sequence_interval=inter_sequence_interval, epochs=epochs)

# Train
epoch_history = manager.run_network_protocol(protocol=protocol, verbose=True, values_to_save_epoch=['w'])

from analysis_functions import calculate_total_connections, calculate_connections_free_attractor_to_first_pattern
from analysis_functions import calculate_connections_last_pattern_to_free_attractor
from analysis_functions import calculate_connections_among_free_attractor
from analysis_functions import calculate_connections_first_pattern_to_free_attractor


# Connections between pattern 0 and pattern 1
print('total_connections 0 to 1', calculate_total_connections(manager, 0, 1, normalize=True))

# Connections between final pattern and free attractor
print('last pattern to free attractor', calculate_connections_last_pattern_to_free_attractor(manager, ampa=False, normalize=True))

# Connections between the free attractor and the first pattern
print('free attractor to first pattern', calculate_connections_free_attractor_to_first_pattern(manager, ampa=False, normalize=True))

# Connections of the free attractor among themselves
print('among free attractor', calculate_connections_among_free_attractor(manager, ampa=False, normalize=True))

print('first pattern to free attractor', calculate_connections_first_pattern_to_free_attractor(manager, ampa=False, normalize=True))

if True:
    traces_to_plot = [0, 6, 1]
    plot_state_variables_vs_time(manager, traces_to_plot, ampa=True)
    plot_state_variables_vs_time(manager, traces_to_plot, ampa=False)
    plot_weight_matrix(nn, ampa=False, one_hypercolum=False)
    plot_weight_matrix(nn, ampa=True, one_hypercolum=False)
    plot_winning_pattern(manager, remove=0.010)
    plot_network_activity_angle(manager)
    plt.show()

# manager.run_network_recall(T_recall=T_recalling, I_cue=patterns[0], T_cue=0.1)
# manager.run_network_recall(T_recall=T_recalling, I_cue=None, T_cue=0.0)

from analysis_functions import calculate_angle_from_history, calculate_winning_pattern_from_distances
from analysis_functions import calculate_patterns_timings, calculate_recall_sucess

pattern = [0, 1, 2, 3, 4]
success_rate = calculate_recall_sucess(manager, T_recalling=T_recalling,
                                      I_cue=patterns[0], T_cue=0.1, n=30, pattern=pattern)

if False:
    plot_weight_matrix(nn, ampa=False, one_hypercolum=False)
    plot_weight_matrix(nn, ampa=True, one_hypercolum=False)
    plot_winning_pattern(manager, remove=0.010)
    plot_network_activity_angle(manager)
    plt.show()