from __future__ import print_function

import pprint

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from network import BCPNN, NetworkManager, BCPNNFast
from data_transformer import build_ortogonal_patterns
from plotting_functions import plot_state_variables_vs_time, plot_network_activity, plot_network_activity_angle
from plotting_functions import  plot_adaptation_dynamics, plot_weight_matrix, plot_winning_pattern, plot_sequence
np.set_printoptions(suppress=True, precision=2)


import seaborn as sns

# Patterns parameters
hypercolumns = 4
minicolumns = 10
n_patterns = 10  # Number of patterns

# Manager properties
dt = 0.001
T_training = 0.5
T_ground = 0.1
T_recalling = 10.0


values_to_save = ['o', 'a', 'z_pre', 'z_post', 'p_pre', 'p_post', 'p_co', 'z_co', 'w',
                  'z_pre_ampa', 'z_post_ampa', 'p_pre_ampa', 'p_post_ampa', 'p_co_ampa', 'z_co_ampa', 'w_ampa']
# values_to_save = ['o']


# Build patterns
patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
patterns = list(patterns_dic.values())

# Build the network
nn = BCPNNFast(hypercolumns, minicolumns)

# Build the manager
manager = NetworkManager(nn=nn, dt=dt, T_training=T_training, T_ground=T_ground, T_recalling=T_recalling,
                         values_to_save=values_to_save)
pprint.pprint(nn.get_parameters())

# Training
repetitions = 1

for i in range(repetitions):
    print('repetitions', i)

    for pattern in patterns:
        nn.k = 1.0
        manager.run_network(time=manager.time_training, I=pattern)

        # Pause between the pattern
        nn.k = 0.0
        manager.run_network(time=manager.time_ground)

manager.T_total = repetitions * (T_training + T_ground) * n_patterns
manager.n_patterns = n_patterns
manager.patterns = patterns

plot_winning_pattern(manager, separators=False, remove=T_training - 0.1)
plot_weight_matrix(nn, ampa=False, one_hypercolum=True)
plot_weight_matrix(nn, ampa=True, one_hypercolum=False)

traces_to_plot = [2, 1, 3]
plot_state_variables_vs_time(manager, traces_to_plot, ampa=False)
plt.show()


# Recalling
if False:
    nn.k = 0.0
    T_cue = 0.25
    time_cue = np.arange(0, T_cue, dt)
    pattern_cue = 5

    nn.reset_values(keep_connectivity=True)

    manager.empty_history()
    manager.run_network(time=time_cue, I=patterns[pattern_cue])
    manager.run_network_recall(reset=False, empty_history=False)

    manager.T_total = 1 * (T_recalling + T_cue)

    plot_winning_pattern(manager, separators=True, remove=0.1)
    plt.show()