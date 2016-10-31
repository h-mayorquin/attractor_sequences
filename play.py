from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from network import BCPNN, NetworkManager
from data_transformer import build_ortogonal_patterns
from connectivity_functions import calculate_coactivations, calculate_probability, get_w, get_beta
from analysis_functions import calculate_angle_from_history
from analysis_functions import calculate_winning_pattern_from_distances, calculate_patterns_timings

np.set_printoptions(suppress=True)

hypercolumns = 10
minicolumns = 10
n_patterns = 10  # Number of patterns

patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
patterns = list(patterns_dic.values())
patterns = patterns[:n_patterns]

# Build the network
nn = BCPNN(hypercolumns, minicolumns)
nn.randomize_pattern()

nn.get_parameters()

dt = 0.001
T_training = 1.0
training_time = np.arange(0, T_training + dt, dt)

values_to_save = ['o', 'z_pre', 'p_pre', 'w', 'a']
manager = NetworkManager(nn=nn, time=training_time, values_to_save=values_to_save)

for pattern in patterns:
    nn.k = 1.0
    print('trained')
    manager.run_network(time=training_time, I=pattern)