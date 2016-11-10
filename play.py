from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from network import BCPNN, NetworkManager
from data_transformer import build_ortogonal_patterns

np.set_printoptions(suppress=True)

hypercolumns = 2
minicolumns = 10
n_patterns = 5  # Number of patterns

patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
patterns = list(patterns_dic.values())
patterns = patterns[:n_patterns]

# Build the network
tau_z_pre = 0.500
tau_z_post = 0.125

nn = BCPNN(hypercolumns, minicolumns, tau_z_post=tau_z_post, tau_z_pre=tau_z_pre)

dt = 0.001
T_training = 1.0
time_training = np.arange(0, T_training, dt)
T_ground = 0.3
time_ground = np.arange(0, T_ground, dt)
values_to_save = ['o', 'z_pre', 'z_post', 'p_pre', 'p_post', 'p_co', 'z_co', 'w']

manager = NetworkManager(nn=nn, values_to_save=values_to_save)

repetitions = 3
resting_state = False
for i in range(repetitions):
    print('repetitions', i)
    for pattern in patterns:
        nn.k = 1.0
        manager.run_network(time=time_training, I=pattern)
        nn.k = 0.0
        if resting_state:
            manager.run_network(time=time_ground)



traces_to_plot = [0, 1, 2]

from plotting_functions import plot_state_variables_vs_time

plot_state_variables_vs_time(manager, n_patterns, T_training, T_ground,
                             repetitions, resting_state, traces_to_plot)
plt.show()