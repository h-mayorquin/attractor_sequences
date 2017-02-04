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
minicolumns = 10
n_patterns = 5  # Number of patterns

# Manager properties
dt = 0.001
T_recalling = 3.0
values_to_save = ['o', 's', 'z_pre', 'z_post', 'p_pre', 'p_post', 'p_co', 'z_co', 'w', 'p', 'k_d']

# Protocol
training_time = 0.1
inter_sequence_interval = 0.5
inter_pulse_interval = 0.2
epochs = 2

# Build patterns
patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
patterns = list(patterns_dic.values())
patterns = patterns[:n_patterns]

# Build the network
nn = BCPNNFast(hypercolumns, minicolumns)

nn.k_inner = True
nn.g_w = 1.0

# Build the manager
manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

# Build the protocol for training
protocol = Protocol()
protocol.simple_protocol(patterns, training_time=training_time, inter_pulse_interval=inter_pulse_interval,
                         inter_sequence_interval=inter_sequence_interval, epochs=epochs)

# Recall
if False:
    # manager.run_network_recall(T_recalling, I_cue=patterns[0], T_cue=0.100)
    manager.run_network_recall(T_recalling)

# Train
if True:
    epoch_history = manager.run_network_protocol(protocol=protocol, verbose=True, values_to_save_epoch=['w'])
    w = epoch_history['w']

# Plot trajectories training
if False:
    traces_to_plot = [1, 0, 2]
    plot_state_variables_vs_time(manager, traces_to_plot, ampa=False)

    traces_to_plot = [0, 5, 1]
    plot_state_variables_vs_time(manager, traces_to_plot, ampa=False)

    traces_to_plot = [6, 5, 7]
    plot_state_variables_vs_time(manager, traces_to_plot, ampa=False)

    traces_to_plot = [6, 0, 7]
    plot_state_variables_vs_time(manager, traces_to_plot, ampa=False)

    traces_to_plot = [5, 4, 6]
    plot_state_variables_vs_time(manager, traces_to_plot, ampa=False)

    plot_weight_matrix(nn, one_hypercolum=True)
    k_d = manager.history['k_d']
    fig = plt.figure(figsize=(16 ,12))
    ax = fig.add_subplot(111)
    ax.plot(k_d)

# Recall
if True:
    manager.run_network_recall(T_recalling, I_cue=patterns[0], T_cue=1.000)
    # manager.run_network_recall(T_recalling)

# Plot trajectories recall
if True:
    traces_to_plot = [1, 0, 2]
    plot_state_variables_vs_time(manager, traces_to_plot, ampa=False)

# Plot patterns that won
if True:
    plot_winning_pattern(manager, ax=None, separators=False, remove=0.010)
    plot_network_activity_angle(manager)
    plot_weight_matrix(nn, one_hypercolum=False)
    plot_weight_matrix(nn, ampa=True, one_hypercolum=False)
    plt.show()

    from analysis_functions import calculate_angle_from_history, calculate_winning_pattern_from_distances
    from analysis_functions import calculate_patterns_timings
    distances = calculate_angle_from_history(manager)
    winning = calculate_winning_pattern_from_distances(distances)
    timings = calculate_patterns_timings(winning, dt, remove=0.01)


if False:
    o = manager.history['o'][:100, 5]
    s = manager.history['s'][:100, :]
    z = manager.history['z_pre'][:100, 5]

    s = manager.history['s'][:, :manager.nn.minicolumns]
    o = manager.history['o'][:, :manager.nn.minicolumns]
    traces_to_plot = [5, 6, 7, 8]

    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for trace in traces_to_plot:
        ax1.plot(s[:, trace])
        ax2.plot(o[:, trace])

    plt.show()

# y.reshape((2, 2, 2, 2)).swapaxes(1, 2).reshape((4, 4)).sum(axis=1)



