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

hypercolumns = 2
minicolumns = 3
n_patterns = 1  # Number of patterns

patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
patterns = list(patterns_dic.values())
patterns = patterns[:n_patterns]

# Build the network
nn = BCPNN(hypercolumns, minicolumns)
nn.randomize_pattern()

dt = 0.001
T_training = 0.003
training_time = np.arange(0, T_training + dt, dt)

values_to_save = ['o', 'z_pre', 'z_post', 'a']
manager = NetworkManager(nn=nn, time=training_time, values_to_save=values_to_save)

for pattern in patterns:
    nn.k = 1.0
    print('trained')
    # history = nn.run_network_simulation(time=training_time, I=pattern, save=True)
    manager.run_network(time=training_time, I=pattern)

history = manager.history
total_time = np.arange(0, n_patterns * (T_training + dt), dt)

z_pre_hypercolum = history['z_pre'][..., :minicolumns]
z_post_hypercolum = history['z_post'][..., :minicolumns]
o_hypercolum = history['o'][..., :minicolumns]
a_hypercolum = history['a'][..., :minicolumns]

# Plot z_traces
fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

for index in range(minicolumns):
    ax1.plot(total_time, o_hypercolum[:, index], label=str(index))
    ax2.plot(total_time, z_pre_hypercolum[:, index], label=str(index))


ax1.legend()
ax2.legend()

ax1.set_ylim([-0.1, 1.1])
ax2.set_ylim([-0.1, 1.1])

plt.show()


