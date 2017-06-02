import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
minicolumns = 20
n_patterns = 10

dt = 0.001

value = 1.0
inhibition = -1
decay_factor = 0.9
sequence_decay = 1.0
extension = 2

sequence_length = 5
overload = 3
overlap = 2
one_to_one = True

# Desired patterns
total_sequences = 10

# Running parameters
max_iter = 1e4

# Random seed
prng = np.random.RandomState(seed=2)
# prng = np.random

overload_range = np.arange(1, minicolumns, 1)
overlap_range = np.arange(0, minicolumns - 1, 1)
recall_success_mean = np.zeros((overload_range.size, overlap_range.size))
recall_success_var = np.zeros((overload_range.size, overlap_range.size))

n = 10
T_cue = 0.100
T_recall = 3.0

sequences = [[0, 1, 2, 3, 4, 5], [3, 6, 7, 8, 9, 10]]
# sequences = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
# sequences = [[0, 1, 2, 3, 4]]


values = np.logspace(-2, 2, 10)

beta_decay = 0.99

intensity = 1.0

from connectivity_functions import artificial_beta_vector
beta = artificial_beta_vector(hypercolumns, minicolumns, sequences, intensity, beta_decay)


if False:
    success_array = np.zeros((values.size, len(sequences)))
    for index, value in enumerate(values):
        manager = create_artificial_manager(hypercolumns, minicolumns, sequences, value, inhibition, extension,
                                            decay_factor,
                                            sequence_decay, dt, BCPNNFast, NetworkManager, ampa=False)

        for n_to_recall in range(len(sequences)):
            sequence_to_recall = sequences[n_to_recall]
            success = calculate_recall_success(manager, T_recall=T_recall, I_cue=sequence_to_recall[0], T_cue=T_cue, n=n,
                                       patterns_indexes=sequence_to_recall)

            success_array[index, n_to_recall] = success

            timings = calculate_timings(manager, remove=0.030)
            print('value', value)
            for x in timings:
                print(x[0])
            print('-------')

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.semilogx(values, success_array[:, 0], '*-', markersize=13)

    ax.set_xlabel('Weight intensity')
    ax.set_xlabel('Recall accuracy')
    plt.show()