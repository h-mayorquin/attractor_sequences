from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from network import BCPNN
from data_transformer import build_ortogonal_patterns
from connectivity_functions import calculate_coactivations, calculate_probability, get_w, get_beta

hypercolumns = 10
minicolumns = 10
N = 10  # Number of patterns

patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
patterns = list(patterns_dic.values())
patterns = patterns[:N]

p = calculate_probability(patterns)
P = calculate_coactivations(patterns)

beta = get_beta(p)
w = get_w(P, p)

dt = 0.001
T = 10.0
time = np.arange(0, T + dt, dt)

nn = BCPNN(hypercolumns, minicolumns, beta=beta, w=w)
nn.randomize_pattern()
history = nn.run_network_simulation(time, save=True)

# Now I need to extract the time series for o
o = history['o']
distances = np.zeros((time. size, len(patterns)))

for index, state in enumerate(o):
    diff = state - patterns
    dis = np.linalg.norm(diff, ord=1, axis=1)
    distances[index] = dis

# Normalize distances
distances = distances / np.sum(distances, axis=1)[:, np.newaxis]
distances = 1 - distances

# Plot everything
if True:
    cmap = 'magma'
    extent = [0, minicolumns * hypercolumns, T, 0]
    fig = plt.figure(figsize=(16 ,12))

    ax1 = fig.add_subplot(121)
    im1 = ax1.imshow(history['o'], aspect='auto', interpolation='None', cmap=cmap, vmax=1, vmin=0, extent=extent)

    ax2 = fig.add_subplot(122)
    im2 = ax2.imshow(distances, aspect='auto', interpolation='None', cmap=cmap, vmax=1, vmin=0, extent=extent)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.12, 0.05, 0.79])
    fig.colorbar(im1, cax=cbar_ax)

    plt.show()