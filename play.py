from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from network import BCPNN, NetworkManager
from data_transformer import build_ortogonal_patterns
from plotting_functions import plot_state_variables_vs_time, plot_network_activity
from plotting_functions import  plot_adaptation_dynamics, plot_weight_matrix

np.set_printoptions(suppress=True)

hypercolumns = 5
minicolumns = 10
n_patterns = 10  # Number of patterns

patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
patterns = list(patterns_dic.values())
patterns = patterns[:n_patterns]

# Build the network
tau_z_pre = 0.500
tau_z_post = 0.125

nn = BCPNN(hypercolumns, minicolumns, tau_z_post=tau_z_post, tau_z_pre=tau_z_pre)

# Manager properties
dt = 0.001
T_training = 1.0
T_ground = 0.5
T_recalling = 10.0
values_to_save = ['o', 'a', 'z_pre', 'z_post', 'p_pre', 'p_post', 'p_co', 'z_co', 'w']
repetitions = 1
resting_state = True
traces_to_plot = [1, 2, 3]

# Build the manager
manager = NetworkManager(nn=nn, dt=dt, T_training=T_training, T_ground=T_ground, T_recalling=T_recalling,
                         repetitions=repetitions, resting_state=resting_state, values_to_save=values_to_save)

# Train the network
manager.run_network_training(patterns)

if True:
    plot_network_activity(manager, manager.calculate_total_training_time(n_patterns))
    plt.show()

    plot_state_variables_vs_time(manager, n_patterns, traces_to_plot)
    plt.show()

    plot_adaptation_dynamics(manager, n_patterns, traces_to_plot, recall=False)
    plt.show()

    plot_weight_matrix(nn)
    plt.show()

# Do the recall
manager.run_network_recall()

if True:
    plot_network_activity(manager)
    plt.show()

    plot_state_variables_vs_time(manager, n_patterns, traces_to_plot, recall=True)
    plt.show()

    plot_adaptation_dynamics(manager, n_patterns, traces_to_plot, recall=True)
    plt.show()
