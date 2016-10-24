from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from network import BCPNN
from data_transformer import build_ortogonal_patterns
from connectivity_functions import calculate_coactivations, calculate_probability, get_w, get_beta

np.set_printoptions(suppress=True)

hypercolumns = 5
minicolumns = 5
N = 5  # Number of patterns

patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
patterns = list(patterns_dic.values())
patterns = patterns[:N]

p = calculate_probability(patterns)
P = calculate_coactivations(patterns)

beta = get_beta(p)
w = get_w(P, p)

dt = 0.001
T = 20.0
time = np.arange(0, T + dt, dt)

nn = BCPNN(hypercolumns, minicolumns, beta=beta, w=w)
nn.randomize_pattern()
history = nn.run_network_simulation(time, save=True)

# Now I need to extract the time series for o
o = history['o']

from analysis_functions import calculate_distance_from_history, calculate_angle_from_history
from analysis_functions import calculate_winning_pattern_from_distances, calculate_patterns_timings

# distances = 1 - calculate_distance_from_history(history, patterns)
distances = calculate_angle_from_history(history, patterns)
winning_patterns = calculate_winning_pattern_from_distances(distances)
patterns_timings = calculate_patterns_timings(winning_patterns, dt)
print(patterns_timings)

# Plot everything
if True:
    cmap = 'magma'
    extent = [0, minicolumns * hypercolumns, T, 0]
    fig = plt.figure(figsize=(16 ,12))

    ax1 = fig.add_subplot(121)
    im1 = ax1.imshow(history['o'], aspect='auto', interpolation='None', cmap=cmap, vmax=1, vmin=0, extent=extent)

    ax2 = fig.add_subplot(122)
    im2 = ax2.imshow(distances, aspect='auto', interpolation='None', cmap=cmap, vmax=1, vmin=0)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.12, 0.05, 0.79])
    fig.colorbar(im1, cax=cbar_ax)

    plt.show()