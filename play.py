import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pprint import pprint

from network import Protocol, BCPNNFast, NetworkManager
from connectivity_functions import artificial_connectivity_matrix, create_indepedent_sequences
from connectivity_functions import create_simple_overlap_sequences
from connectivity_functions import test_overload_criteria, remove_overloaded_indexes
from connectivity_functions import modify_overload_matrix
from connectivity_functions import test_overlap_criteria
from connectivity_functions import modify_overlap_dictionary

import pdb

# Patterns parameters
hypercolumns = 4
minicolumns = 25
n_patterns = 10

value = 1.0
inhibition = -1
decay_factor = 1.0
sequence_decay = 1.0
extension = 2

sequence_length = 4
overload = 6
overlap = 3
one_to_one = False

# Auxiliary structures
sequences = []
overload_matrix = np.zeros(minicolumns)
available = [i for i in range(minicolumns)]
removed = []
overlap_dictionary = {}

# Desired patterns
total_sequences = 10

# Running parameters
max_iter = 1e5
iter = 0
n_sequence = 0

# pdb.set_trace()
while n_sequence < total_sequences and iter < max_iter:
    iter += 1

    # Generate a possible sample
    if len(available) >= sequence_length:
        sample = np.random.choice(available, size=sequence_length, replace=False)
    else:
        break

    # Criteria for overload
    overload_criteria = test_overload_criteria(sample, overload_matrix, overload)

    # Criteria for overlap
    candidate_overlap = np.zeros(minicolumns)
    overlap_criteria = test_overlap_criteria(sample, sequences, overlap_dictionary, overlap, candidate_overlap, one_to_one)

    if overlap_criteria and overload_criteria:
        # Add the sample
        sample_list = list(sample.copy())
        sample_list.sort()
        sequences.append(sample_list)

        modify_overlap_dictionary(overlap_dictionary, candidate_overlap, sample, n_sequence, sequences)
        modify_overload_matrix(sample, overload_matrix)
        remove_overloaded_indexes(overload_matrix, overload, available, removed)

        n_sequence += 1

# Plot sequences
def plot_sequences(sequences, minicolumns):
    sequence_matrix = np.zeros((len(sequences), minicolumns))
    for index, sequence in enumerate(sequences):
        sequence_matrix[index, sequence] = index + 1

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)

    cmap = matplotlib.cm.Paired
    cmap.set_under('white')

    ax.imshow(sequence_matrix, cmap=cmap, vmin=0.5)

overlap_matrix = np.zeros((len(sequences), len(sequences)))

for index_1, sequence_1 in enumerate(sequences):
    for index_2, sequence_2 in enumerate(sequences):
        intersection = [val for val in sequence_1 if val in sequence_2]
        overlap_matrix[index_1, index_2] = len(intersection)

print(overload_matrix)
print(overlap_matrix)
plot_sequences(sequences, minicolumns)
plt.show()

