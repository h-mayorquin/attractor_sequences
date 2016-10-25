from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from network import BCPNN
from data_transformer import build_ortogonal_patterns
from connectivity_functions import calculate_coactivations, calculate_probability, get_w, get_beta
from analysis_functions import calculate_angle_from_history
from analysis_functions import calculate_winning_pattern_from_distances, calculate_patterns_timings

np.set_printoptions(suppress=True)

hypercolumns = 10
minicolumns = 10
n_patterns = 5  # Number of patterns

patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
patterns = list(patterns_dic.values())
patterns = patterns[:n_patterns]

dt = 0.001
T_training = 1.0
T_simulation = 10.0
training_time = np.arange(0, T_training + dt, dt)
simulation_time = np.arange(0, T_simulation + dt, dt)

nn = BCPNN(hypercolumns, minicolumns, g_I=10.0)
nn.k = 1.0
nn.randomize_pattern()

for pattern in patterns:
    history = nn.run_network_simulation(training_time, I=pattern, save=True)
# Now I need to extract the time series for o

o = history['o']


# distances = 1 - calculate_distance_from_history(history, patterns)
distances = calculate_angle_from_history(history, patterns)
winning_patterns = calculate_winning_pattern_from_distances(distances)
patterns_timings = calculate_patterns_timings(winning_patterns, dt)
print(patterns_timings)

# Plot everything
if True:
    cmap = 'magma'
    extent = [0, minicolumns * hypercolumns, n_patterns * T_training, 0]
    extent2 = [0, minicolumns * hypercolumns, n_patterns, 0]
    fig = plt.figure(figsize=(16 ,12))

    ax1 = fig.add_subplot(121)
    im1 = ax1.imshow(history['o'], aspect='auto', interpolation='None', cmap=cmap, vmax=1, vmin=0, extent=extent)

    ax2 = fig.add_subplot(122)
    im2 = ax2.imshow(history['a'], aspect='auto', interpolation='None', cmap=cmap, vmax=1, vmin=0, extent=extent)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.12, 0.05, 0.79])
    fig.colorbar(im1, cax=cbar_ax)

    plt.show()