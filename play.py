import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pprint import pprint

from network import Protocol, BCPNNFast, NetworkManager
from connectivity_functions import calculate_overlap_one_to_all, calculate_overlap_one_to_one
from connectivity_functions import  calculate_random_sequence, calculate_overlap_matrix
from plotting_functions import plot_artificial_sequences
import pdb


# Patterns parameters
hypercolumns = 4
minicolumns = 15
n_patterns = 10

value = 1.0
inhibition = -1
decay_factor = 1.0
sequence_decay = 1.0
extension = 2

sequence_length = 4
overload = 5
overlap = 2
one_to_one = True


# Desired patterns
total_sequences = 20

# Running parameters
max_iter = 1e4

# Random seed
prng = np.random.RandomState(seed=2)
# pdb.set_trace()

aux = calculate_random_sequence(minicolumns, sequence_length, overlap,  overload,  one_to_one=one_to_one,
                                prng=prng, total_sequences=total_sequences, max_iter=max_iter)
sequences, overlap_dictionary, overload_matrix = aux

overlap_matrix = calculate_overlap_matrix(sequences)



print('overload matrix')
print(overload_matrix)
print('overlap matrix')
print(overlap_matrix)


plot_artificial_sequences(sequences, minicolumns)
plt.show()

