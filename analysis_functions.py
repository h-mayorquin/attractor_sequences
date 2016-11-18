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
    history = manager.history
    patterns = manager.patterns
    o = history['o']

    distances = np.zeros((o.shape[0], manager.nn.minicolumns))

    for index, state in enumerate(o):
        nominator = [np.dot(state, pattern) for pattern in patterns]
        denominator = [np.linalg.norm(state) * np.linalg.norm(pattern) for pattern in patterns]

        dis = [a / b for (a, b) in zip(nominator, denominator)]
        distances[index, :manager.nn.minicolumns] = dis

    return distances


def calculate_winning_pattern_from_distances(distances):

    return np.argmax(distances, axis=1)


def calculate_patterns_timings(winning_patterns, dt, remove=0):

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

