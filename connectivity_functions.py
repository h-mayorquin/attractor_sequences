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
