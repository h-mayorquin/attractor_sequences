import numpy as np
import IPython


def log_epsilon(x, epsilon=1e-10):

    return np.log(np.maximum(x, epsilon))


def calculate_probability(patterns):
    """
    Returns the probability from a list of patterns to be learned
    :param patterns: list of patterns to be learned
    :return:
    """

    p = np.zeros(patterns[0].size)
    number_of_patterns = len(patterns)

    for pattern in patterns:
        p += pattern

    p /= number_of_patterns

    return p


def calculate_coactivations(patterns):

    coactivations = np.zeros((patterns[0].size, patterns[0].size))
    number_of_patterns = len(patterns)

    for pattern in patterns:
        coactivations += np.outer(pattern, pattern)

    coactivations /= number_of_patterns

    return coactivations


def get_w(P, p, diagonal_zero=True):

    outer = np.outer(p, p)

    w = log_epsilon(P) - log_epsilon(outer)
    if diagonal_zero:
        w[np.diag_indices_from(w)] = 0

    return w


def get_w_pre_post(P, p_pre, p_post, p=1.0, epsilon=1e-20, diagonal_zero=True):

    outer = np.outer(p_post, p_pre)

    # w = np.log(p * P) - np.log(outer)
    x = p * (P / outer)
    # w = np.log(x)
    w = log_epsilon(x, epsilon)

    if diagonal_zero:
        w[np.diag_indices_from(w)] = 0

    return w


def get_beta(p, epsilon=1e-10):

    probability = np.copy(p)
    probability[p < epsilon] = epsilon

    beta = np.log(probability)

    return beta


def softmax(x, t=1.0, minicolumns=2):
    """Calculate the softmax of a list of numbers w.

    Parameters
    ----------
    w : list of numbers
    t : float

    Return
    ------
    a list of the same length as w of non-negative numbers

    Examples
    --------
    >>> softmax([0.1, 0.2])

    array([ 0.47502081,  0.52497919])
    >>> softmax([-0.1, 0.2])

    array([ 0.42555748,  0.57444252])
    >>> softmax([0.9, -10])

    array([  9.99981542e-01,   1.84578933e-05])
    >>> softmax([0, 10])
    array([  4.53978687e-05,   9.99954602e-01])
    """
    x_size = x.size
    x = np.reshape(x, (x_size / minicolumns, minicolumns))

    e = np.exp(np.array(x) / t)
    dist = normalize_array(e)

    dist = np.reshape(dist, x_size)
    return dist


def normalize_array(array):

    return array / np.sum(array, axis=1)[:, np.newaxis]


def normalize_p(p, hypercolumns, minicolumns):

    x = p.reshape((hypercolumns, minicolumns))
    x = x / np.sum(x, axis=1)[:, np.newaxis]

    return x.reshape(hypercolumns * minicolumns)


def load_minicolumn_matrix(w, sequence_indexes, value=1, extension=1, decay_factor=1.0, sequence_decay=1.0):

    n_patterns = len(sequence_indexes)

    for index, pattern_index in enumerate(sequence_indexes[:-1]):
        # Determine the value to load
        sequence_value = value * (sequence_decay ** index)

        # First we set the the sequence connection
        from_unit = pattern_index
        to_unit = sequence_indexes[index + 1]
        w[to_unit, from_unit] = sequence_value

        # Then set the after-effects (extension)
        if index < n_patterns - extension - 1:
            aux = extension
        else:
            aux = n_patterns - index - 1

        for j in range(aux):
            to_unit = sequence_indexes[index + 1 + j]
            w[to_unit, from_unit] = sequence_value * (decay_factor ** j)


def load_diagonal(w, sequence_index, value=1.0):
    for index, pattern_index in enumerate(sequence_index):
        w[pattern_index, pattern_index] = value

def expand_matrix(w_small, hypercolumns, minicolumns):

    w_big = np.zeros((minicolumns * hypercolumns, minicolumns * hypercolumns))
    for j in range(hypercolumns):
        for i in range(hypercolumns):
            w_big[i * minicolumns:(i + 1) * minicolumns, j * minicolumns:(j + 1) * minicolumns] = w_small

    return w_big


def artificial_connectivity_matrix(hypercolumns, minicolumns, sequences, value=1, inhibition=-1, extension=1,
                                   decay_factor=1.0, sequence_decay=1.0, diagonal_zero=True, self_influence=True,
                                   ampa=False):

    w = np.ones((minicolumns, minicolumns)) * inhibition

    if self_influence:
        for sequence_indexes in sequences:
            load_diagonal(w, sequence_indexes, value)

    if not ampa:
        for sequence_indexes in sequences:
            load_minicolumn_matrix(w, sequence_indexes, value, extension, decay_factor, sequence_decay)

    # Create the big matrix
    w_big = expand_matrix(w, hypercolumns, minicolumns)

    # Remove diagonal
    if diagonal_zero:
        w_big[np.diag_indices_from(w_big)] = 0

    return w_big


def create_indepedent_sequences(minicolumns, sequence_length):
    n_sequences = minicolumns / sequence_length
    sequences = [[j*sequence_length + i for i in range(sequence_length)] for j in range(n_sequences)]

    return sequences


def create_simple_overlap_sequences(minicolumns, sequence_length, overlap):
    sequences = []
    increase = sequence_length - overlap
    starting_point = 0
    while starting_point + sequence_length <= minicolumns:
        sequences.append([starting_point + i for i in range(sequence_length)])
        starting_point += increase

    return sequences


# The functions for generating sequences
def test_overload_criteria(sample, overload_matrix, overload):
    criteria = False
    if np.all(overload_matrix[sample] < overload):
        criteria = True
    return criteria


def modify_overload_matrix(sample, overload_matrix):
    overload_matrix[sample] += 1


def remove_overloaded_indexes(overload_matrix, overload, available, removed):
    # Take out the numbers who are overload enough
    indexes_to_remove = np.where(overload_matrix >= overload)[0]
    for index in indexes_to_remove:
        if index not in removed:
            available.remove(index)
            removed.append(index)


def test_overlap_criteria(sample, sequences, overlap_dictionary, overlap, candidate_overlap, one_to_one):
    """
    Test whether the new sample is not in violation of the overlap criteria
    :param sample:
    :param sequences:
    :param overlap_dictionary:
    :param overlap:
    :param candidate_overlap:
    :param one_to_one:
    :return: overlap_criteria
    """

    overlap_criteria = True

    for sequence_number, overlap_vector in overlap_dictionary.items():
        # Intersection
        intersection = [val for val in sample if val in sequences[sequence_number]]

        # If intersection is greater than overlap than overlap then change criteria
        candidate_overlap[intersection] = 1

        if one_to_one:
            if len(intersection) > overlap:
                overlap_criteria = False
                break

        # I have not figure out what this does, apparently it selects for overlap with the same units
#        else:
#            if np.sum(candidate_overlap) > overlap:
#                overlap_criteria = False
#                break

    if not one_to_one:
        for sequence_number, overlap_vector in overlap_dictionary.items():
            intersection = [val for val in sample if val in sequences[sequence_number]]
            if len(intersection) + np.sum(overlap_vector) > overlap:
                overlap_criteria = False

    return overlap_criteria


def modify_overlap_dictionary(overlap_dictionary, candidate_overlap, sample, n_sequence, sequences):
    """
    This modifies the dictionary once a particular sample has been accepted in the sequences

    :param overlap_dictionary: The dictionary with over
    :param candidate_overlap:
    :param sample:
    :param n_sequence:
    :param sequences:
    :return:
    """
    for sequence_number, overlap_vector in overlap_dictionary.items():
        intersection = [val for val in sample if val in sequences[sequence_number]]
        overlap_vector[intersection] = 1

    # Insert the overlap_candidate
    overlap_dictionary[n_sequence] = candidate_overlap


def remove_overlaped_indexes(overlap_dictionary, sequences, overlap, available, removed):
    for sequence_number, overlap_vector in overlap_dictionary.items():
        if np.sum(overlap_vector) >= overlap:
            indexes_to_remove = sequences[sequence_number]
            for index in indexes_to_remove:
                if index not in removed:
                    available.remove(index)
                    removed.append(index)


def calculate_overlap_matrix(sequences):
    overlap_matrix = np.zeros((len(sequences), len(sequences)))
    for index_1, sequence_1 in enumerate(sequences):
        for index_2, sequence_2 in enumerate(sequences):
            intersection = [val for val in sequence_1 if val in sequence_2]
            overlap_matrix[index_1, index_2] = len(intersection)

    overlap_matrix[np.diag_indices_from(overlap_matrix)] = 0

    return overlap_matrix

################
# Old functions
#################

# def get_w_old(P, p, diagonal_zero=True):
#     outer = np.outer(p, p)
#     P_copy = np.copy(P)
#
#     outer[outer < epsilon**2] = epsilon**2
#     P_copy[P < epsilon] = epsilon**2
#
#     w = np.log(P_copy / outer)
#
#     #IPython.embed()
#     if diagonal_zero:
#         w[np.diag_indices_from(w)] = 0
#     return w
#
#
# def get_w_protocol1(P, p):
#     p_copy = np.copy(p)
#     P_copy = np.copy(P)
#
#     p_copy[p < epsilon] = epsilon
#     P_copy[P < epsilon] = epsilon * epsilon
#
#     aux = np.outer(p_copy, p_copy)
#     w = np.log(P_copy / aux)
#     # IPython.embed()
#
#     return w
