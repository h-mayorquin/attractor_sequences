import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pprint import pprint

from network import Protocol, BCPNNFast, NetworkManager
from connectivity_functions import  artificial_connectivity_matrix
from connectivity_functions import calculate_overlap_one_to_all, calculate_overlap_one_to_one
from connectivity_functions import  calculate_random_sequence, calculate_overlap_matrix
from plotting_functions import plot_artificial_sequences
from plotting_functions import plot_winning_pattern
from analysis_functions import calculate_timings, calculate_recall_success
from plotting_functions import plot_weight_matrix

from analysis_functions import calculate_recall_success_sequences
from connectivity_functions import create_artificial_manager

import pdb
# pdb.set_trace()

# Patterns parameters
hypercolumns = 4
minicolumns = 15
n_patterns = 10

dt = 0.001

value = 1.0
inhibition = -1
decay_factor = 0.9
sequence_decay = 0.9
extension = 2

sequence_length = 4
overload = 3
overlap = 3
one_to_one = True

# Desired patterns
total_sequences = 1000

# Running parameters
max_iter = 1e4

# Random seed
prng = np.random.RandomState(seed=2)
prng = np.random

overload_range = np.arange(1, 200, 20)
overlap_range = np.arange(1, 200, 20)
n_sequences_mean = np.zeros((overload_range.size, overlap_range.size))
n_sequences_var = np.zeros((overload_range.size, overlap_range.size))
n_calculations = 10


for index_overload, overload in enumerate(overload_range):
    print(index_overload)
    for index_overlap, overlap in enumerate(overlap_range):
        n_list = []
        for i in range(n_calculations):
            aux = calculate_random_sequence(minicolumns, sequence_length, overlap,  overload,  one_to_one=one_to_one,
                                            prng=prng, total_sequences=total_sequences, max_iter=max_iter)

            sequences, overlap_dictionary, overload_matrix = aux
            n_sequences = len(sequences)
            n_list.append(n_sequences)

        n_sequences_mean[index_overload, index_overlap] = np.mean(n_list)
        n_sequences_var[index_overload, index_overlap] = np.var(n_list)


from mpl_toolkits.axes_grid1 import make_axes_locatable
fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

extent = [overlap_range[0], overlap_range[-1], overload_range[0], overload_range[-1]]

cmap = 'inferno'
im1 = ax1.imshow(n_sequences_mean, origin='lower', cmap=cmap, interpolation='None', extent=extent)
im2 = ax2.imshow(n_sequences_var, origin='lower', cmap=cmap, interpolation='None', extent=extent)

divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax1, orientation='vertical')

divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax2, orientation='vertical')

ax1.set_xlabel('Overlap')
ax1.set_ylabel('Overload')
ax2.set_xlabel('Overlap')
ax2.set_ylabel('Overload')

plt.show()

if False:
    manager = create_artificial_manager(hypercolumns, minicolumns, sequences, value, inhibition, extension, decay_factor,
                                        sequence_decay, dt, BCPNNFast, NetworkManager)

    # Recall
    n = 10
    T_cue = 0.100
    T_recall = 2.0
    successes = calculate_recall_success_sequences(manager, T_recall, T_cue, n, sequences)
