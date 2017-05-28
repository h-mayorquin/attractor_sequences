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
from analysis_functions import calculate_timings

import pdb
# pdb.set_trace()

# Patterns parameters
hypercolumns = 4
minicolumns = 50
n_patterns = 10

dt = 0.001

value = 1.0
inhibition = -1
decay_factor = 1.0
sequence_decay = 1.0
extension = 2

sequence_length = 5
overload = 2
overlap = 4
one_to_one = True


# Desired patterns
total_sequences = 2

# Running parameters
max_iter = 1e4

# Random seed
prng = np.random.RandomState(seed=2)


aux = calculate_random_sequence(minicolumns, sequence_length, overlap,  overload,  one_to_one=one_to_one,
                                prng=prng, total_sequences=total_sequences, max_iter=max_iter)
sequences, overlap_dictionary, overload_matrix = aux
n_sequences = len(sequences)

w_nmda = artificial_connectivity_matrix(hypercolumns, minicolumns, sequences, value=value, inhibition=inhibition,
                                        extension=extension, decay_factor=decay_factor, sequence_decay=sequence_decay,
                                        diagonal_zero=True, self_influence=True, ampa=False)

w_ampa = artificial_connectivity_matrix(hypercolumns, minicolumns, sequences, value=value, inhibition=inhibition,
                                        extension=extension, decay_factor=decay_factor, sequence_decay=sequence_decay,
                                        diagonal_zero=True, self_influence=True, ampa=True)

nn = BCPNNFast(hypercolumns=hypercolumns, minicolumns=minicolumns)
nn.w = w_nmda
nn.w_ampa = w_ampa
manager = NetworkManager(nn, dt=dt, values_to_save=['o'])
for pattern_indexes in sequences:
    manager.stored_patterns_indexes += pattern_indexes


# Recall
T_cue = 0.100
n_recall = 0
sequence_to_recall = sequences[n_recall]
print('sequence_to_Recall')
print(sequence_to_recall)
I_cue = sequence_to_recall[0]
T_recall = 2.0
manager.run_network_recall(T_recall=T_recall, I_cue=I_cue, T_cue=T_cue)
plot_winning_pattern(manager)
timings = calculate_timings(manager, remove=0.010)
pair = [(x[0], x[3]) for x in timings]
print(pair)
plt.show()