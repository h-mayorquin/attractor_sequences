import numpy as np


def calculate_distance_from_history(history, patterns, normalize=True):

    o = history['o']
    distances = np.zeros((o.shape[0], len(patterns)))

    for index, state in enumerate(o):
        diff = state - patterns
        dis = np.linalg.norm(diff, ord=1, axis=1)
        distances[index] = dis

    if normalize:
        distances = distances / np.sum(distances, axis=1)[:, np.newaxis]
    return distances


def calculate_angle_from_history(manager):
    """
    :param manager: A manager of neural networks, it is used to obtain the history of the activity and
     the patterns that were stored

    :return: A vector with the distances to the stored patterns. This vector will be as long as the number of points
     in time times the number of pattern stores
    """
    history = manager.history
    patterns_dic = manager.patterns_dic
    stored_pattern_indexes = manager.stored_patterns_indexes

    o = history['o']

    distances = np.zeros((o.shape[0], manager.nn.minicolumns))

    for index, state in enumerate(o):
        # Obtain the dot product between the state of the network at each point in time and each pattern
        nominator = [np.dot(state, patterns_dic[pattern_index]) for pattern_index in stored_pattern_indexes]
        # Obtain the norm of both the state and the patterns to normalize
        denominator = [np.linalg.norm(state) * np.linalg.norm(patterns_dic[pattern_index])
                       for pattern_index in stored_pattern_indexes]

        # Get the angles and store them
        dis = [a / b for (a, b) in zip(nominator, denominator)]
        distances[index, stored_pattern_indexes] = dis

    return distances


def calculate_winning_pattern_from_distances(distances):
    # Returns the number of the winning pattern
    return np.argmax(distances, axis=1)


def calculate_patterns_timings(winning_patterns, dt, remove=0):
    """

    :param winning_patterns: A vector with the winning pattern for each point in time
    :param dt: the amount that the time moves at each step
    :param remove: only add the patterns if they are bigger than this number, used a small number to remove fluctuations

    :return: pattern_timins, a vector with information about the winning pattern, how long the network stayed at that
     configuration, when it got there, etc
    """

    # First we calculate where the change of pattern occurs
    change = np.diff(winning_patterns)
    indexes = np.where(change != 0)[0]

    # Add the end of the sequence
    indexes = np.append(indexes, winning_patterns.size - 1)

    patterns = winning_patterns[indexes]
    patterns_timings = []

    previous = 0
    for pattern, index in zip(patterns, indexes):
        time = (index - previous + 1) * dt  # The one is because of the shift with np.change
        if time >= remove:
            patterns_timings.append((pattern, time, previous*dt, index * dt))
        previous = index

    return patterns_timings


def calculate_recall_success(manager, T_recall,  I_cue, T_cue, n, patterns_indexes):

    n_patterns = manager.n_patterns
    successes = 0
    for i in range(n):
        manager.run_network_recall(T_recall=T_recall, I_cue=I_cue, T_cue=T_cue)

        distances = calculate_angle_from_history(manager)
        winning = calculate_winning_pattern_from_distances(distances)
        timings = calculate_patterns_timings(winning, manager.dt, remove=0)
        pattern_sequence = [x[0] for x in timings]

        if pattern_sequence[:n_patterns] == patterns_indexes:
            successes += 1

    success_rate = successes * 100.0/ n

    return success_rate


# Functions to extract connectivity
def calculate_total_connections(manager, from_pattern, to_pattern, ampa=False, normalize=True):

    if ampa:
        w = manager.nn.w_ampa
    else:
        w = manager.nn.w

    hypercolumns = manager.nn.hypercolumns
    minicolumns = manager.nn.minicolumns

    from_pattern_j = from_pattern
    to_pattern_i = to_pattern

    weights = 0.0
    pattern_i_indexes = [int(to_pattern_i + i * minicolumns) for i in range(hypercolumns)]
    pattern_j_indexes = [int(from_pattern_j + j * minicolumns) for j in range(hypercolumns)]

    for j_index in pattern_j_indexes:
        weights += w[pattern_i_indexes, j_index].sum()

    norm = (hypercolumns * hypercolumns)
    if normalize:
        weights /= norm

    return weights


def calculate_connections_last_pattern_to_free_attractor(manager, ampa=False, normalize=True):
    if ampa:
        w = manager.nn.w_ampa
    else:
        w = manager.nn.w

    n_patterns = manager.n_patterns
    minicolumns = manager.nn.minicolumns

    final_pattern = n_patterns - 1
    free_attractor_indexes = np.arange(n_patterns, minicolumns, dtype='int')
    weights = w[free_attractor_indexes, final_pattern].sum()

    norm = len(free_attractor_indexes)
    if normalize:
        weights /= norm

    return weights


def calculate_connections_free_attractor_to_first_pattern(manager, ampa=False, normalize=True):
    if ampa:
        w = manager.nn.w_ampa
    else:
        w = manager.nn.w

    n_patterns = manager.n_patterns
    minicolumns = manager.nn.minicolumns

    first_pattern = 0
    free_attractor_indexes = np.arange(n_patterns, minicolumns, dtype='int')
    weights = w[first_pattern, free_attractor_indexes].sum()

    norm = len(free_attractor_indexes)
    if normalize:
        weights /= norm

    return weights


def calculate_connections_first_pattern_to_free_attractor(manager, ampa=False, normalize=True):
    if ampa:
        w = manager.nn.w_ampa
    else:
        w = manager.nn.w

    n_patterns = manager.n_patterns
    minicolumns = manager.nn.minicolumns

    first_pattern = 0
    free_attractor_indexes = np.arange(n_patterns, minicolumns, dtype='int')
    weights = w[free_attractor_indexes, first_pattern].sum()

    norm = len(free_attractor_indexes)
    if normalize:
        weights /= norm

    return weights


def calculate_connections_among_free_attractor(manager, ampa=False, normalize=True):
    if ampa:
        w = manager.nn.w_ampa
    else:
        w = manager.nn.w

    n_patterns = manager.n_patterns
    minicolumns = manager.nn.minicolumns

    weights = w[n_patterns:minicolumns, n_patterns:minicolumns].sum()

    norm = (minicolumns - n_patterns) ** 2
    if normalize:
        weights /= norm

    return weights


