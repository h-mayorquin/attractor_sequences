import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

from network import Protocol, BCPNNFast, NetworkManager
from plotting_functions import plot_winning_pattern
from analysis_functions import calculate_recall_success, calculate_timings
from analysis_functions import create_artificial_matrix
from plotting_functions import plot_weight_matrix, plot_network_activity_angle
from analysis_functions import calculate_angle_from_history, calculate_winning_pattern_from_distances


# Network parameters
minicolumns = 60
hypercolumns = 4
number_of_patterns = 30
patterns_indexes = [i for i in range(number_of_patterns)]

# Training parameters
dt = 0.001
value = 1.0
extension = 1
decay_factor = 0.1
diagonal_zero = True
diagonal_across = True
diagonal_value = 1.2
inhibition = -1
sequence_decay = 0.99

# Recall
T_cue = 1.0
I_cue = 0
T_recall = 5.0
value_ranges = np.arange(0.1, 3, 0.6)
mean_recall_time = []
std_recall_time = []


w = create_artificial_matrix(hypercolumns, minicolumns, number_of_patterns, value, inhibition, decay_factor,
                         extension, diagonal_zero, diagonal_across, diagonal_value, sequence_decay=sequence_decay)
w_ampa = create_artificial_matrix(hypercolumns, minicolumns, number_of_patterns, value, inhibition, decay_factor,
                         0, diagonal_zero, diagonal_across, diagonal_value)

# Create the network
nn = BCPNNFast(hypercolumns=hypercolumns, minicolumns=minicolumns)
nn.w = w
nn.w_ampa = w_ampa
manager = NetworkManager(nn, dt=dt, values_to_save=['o'])
manager.stored_patterns_indexes = patterns_indexes
manager.n_patterns = number_of_patterns

plot_weight_matrix(nn, ampa=False, one_hypercolum=True)
plt.show()

if False:
    manager.run_network_recall(T_recall=T_recall, I_cue=I_cue, T_cue=T_cue)

    timings = calculate_timings(manager)
    timings = [timings[index][1] for index in patterns_indexes[1:-1]]
    mean_recall_time.append(np.mean(timings))
    std_recall_time.append(np.std(timings))



