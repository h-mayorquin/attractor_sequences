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
T_recalling = 5.0
values_to_save = ['o', 's', 'z_pre', 'z_post', 'p_pre', 'p_post', 'p_co', 'z_co', 'w']

# Protocol
training_time = 0.1
inter_sequence_interval = 2.0
inter_pulse_interval = 0.0
epochs = 3

# Build patterns
patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
patterns = list(patterns_dic.values())
# patterns = patterns[:n_patterns]

# Build the network
nn = BCPNNFast(hypercolumns, minicolumns)

nn.k_inner = False


# Build the manager
manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

# Build the protocol for training
# patterns1 = [patterns[0], patterns[1], patterns[10], patterns[11], patterns[12]]
# patterns2 = [patterns[2], patterns[3], patterns[10], patterns[13], patterns[14]]

patterns1 = [patterns[0], patterns[1], patterns[2], patterns[3], patterns[4]]
patterns2 = [patterns[10], patterns[11], patterns[12], patterns[13], patterns[14]]

epochs = 1
chain = [patterns1, patterns2]

protocol = Protocol()
protocol.cross_protocol(chain, training_time=training_time,
                        inter_sequence_interval=inter_sequence_interval, epochs=epochs)

# Train
epoch_history = manager.run_network_protocol(protocol=protocol, verbose=True, values_to_save_epoch=['w'])
plot_weight_matrix(nn, ampa=True, one_hypercolum=True)
plot_weight_matrix(nn, ampa=False, one_hypercolum=True)
plot_network_activity_angle(manager)
plot_winning_pattern(manager, remove=0.1)
plt.show()







