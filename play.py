from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from network import BCPNN, NetworkManager, BCPNNFast
from data_transformer import build_ortogonal_patterns
from plotting_functions import plot_state_variables_vs_time, plot_network_activity, plot_network_activity_angle
from plotting_functions import  plot_adaptation_dynamics, plot_weight_matrix

np.set_printoptions(suppress=True)

# Patterns parameters
hypercolumns = 4
minicolumns = 10
n_patterns = 10  # Number of patterns

# Network parameters
tau_z_pre = 0.500
tau_z_post = 0.125
tau_z_pre_ampa = 0.005
tau_z_post_ampa = 0.005
tau_a = 2.7

# Manager properties
dt = 0.001
T_training = 1.0
T_ground = 1.0
T_recalling = 50
values_to_save = ['o', 'a', 'z_pre', 'z_post', 'p_pre', 'p_post', 'p_co', 'z_co', 'w',
                  'z_pre_ampa', 'z_post_ampa', 'p_pre_ampa', 'p_post_ampa', 'p_co_ampa', 'z_co_ampa', 'w_ampa']

repetitions = 1
resting_state = False
traces_to_plot = [2, 3, 4]

# Build patterns
patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
patterns = list(patterns_dic.values())
patterns = patterns[:n_patterns]

# Build the network
nn = BCPNNFast(hypercolumns, minicolumns, tau_z_post=tau_z_post, tau_z_pre=tau_z_pre,
               tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa)

# Build the manager
manager = NetworkManager(nn=nn, dt=dt, T_training=T_training, T_ground=T_ground, T_recalling=T_recalling,
                         repetitions=repetitions, resting_state=resting_state, values_to_save=values_to_save)

for pattern in patterns[:6]:
    nn.k = 1.0
    manager.run_network(time=manager.time_training, I=pattern)
    nn.k = 0.0
    manager.run_network(time=manager.time_ground, I=None)

for pattern in patterns[6:]:
    nn.k = 1.0
    manager.run_network(time=manager.time_training, I=pattern)

manager.n_patterns = n_patterns
manager.T_total = 6 * (T_training + T_ground) + 4 * T_training

plot_network_activity(manager)
plot_adaptation_dynamics(manager, traces_to_plot)

manager.run_network_recall()
manager.patterns = patterns
plot_network_activity_angle(manager)
plt.show()