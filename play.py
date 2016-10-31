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

hypercolumns = 3
minicolumns = 3
n_patterns = 3  # Number of patterns

patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
patterns = list(patterns_dic.values())
patterns = patterns[:n_patterns]

dt = 0.01
T = 1.0
time = np.arange(0, T + dt, dt)
saving_dictionary = {'o': True, 's': True, 'z_pre': False,
                     'z_post': False, 'a': True, 'p_pre': False,
                     'p_post': False, 'p_co': False, 'w': False,
                     'beta': False}

nn = BCPNN(hypercolumns, minicolumns)
manager = NetworkManager(nn=nn, time=time, saving_dictionary=saving_dictionary)
history = manager.run_network(save=True, I=patterns[1])


