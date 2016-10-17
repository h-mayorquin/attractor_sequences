from __future__ import print_function
import numpy as np


def calculate_arg_min_list(list):
    """
    Calculates the argument of the minimum in a list
    :param list:
    :return:the argument of the minimum
    """
    return list.index(min(list))


def calculate_distances_to_fix_points_list(point, patterns):
    """
    This calculates the distances of a point to all the patterns and
    returns the distances in a list
    :param point: a point in the space of interest
    :param patterns: the collection of the fixed points
    :return: a list with the distances to all the fixed points
    """

    return [np.linalg.norm(point - pattern) for pattern in patterns]


def calculate_closest_pattern_dictionary(distances_dictionary):
    """
    Calculates the closets pattern from a dictionary with the distances of
    a point to a dictionary
    :param distances_dictionary: a dictionary where each key is the index of a
     fis point and the value is the distance to that fix point
    :return:the pattern with the smallest distance
    """

    return min(distances_dictionary, key=distances_dictionary.get)


def calculate_distances_to_fix_points_dictionary(point, patterns):
    """
    Calculates the distnace a point to a list of fix points and returns a
    dictionary where the key is the index of the fix point and the value
    is the distance between that point and the fixed point
    :param point: just a random point int he space of interest
    :param patterns: a list iwth all the fixed patterns
    :return: The dictionary
    """

    return {pattern_number: np.linalg.norm(point - pattern) for pattern_number, pattern in enumerate(patterns)}


def append_distances_history(point, patterns, closest_pattern,
                           distances_history, save_distances=True):
    """

    Appends the history of the distances to the appropriate lists
    :param point: The point in the history
    :param patterns:
    :param closest_pattern:
    :param distances_history:
    :param save_distances:
    :return:
    """

    distances_dic = calculate_distances_to_fix_points_dictionary(point, patterns)
    closest_pattern.append(calculate_closest_pattern_dictionary(distances_dic))

    if save_distances:
        distances_history.append(distances_dic)


def calculate_convergence_ratios(nn, N, time, patterns):
    """
    This funtion is used to test two things.

    First, the number of times that a random initiation of the neural network (already trained) ends up
    in one of the patterns stored in it. We call this fraction nof convergence and is normalized by
    the number of runs.

    Second, the nuumber of times that an initial patterns ends up settling in the point thaat it was
    closest to at the beginning of the simulation. We call this the fraction of well behaviour and
    is normalized as well by the bnumber of runs

    :param nn: An instances of a BCPNN
    :param N:  The number of runs to use
    :param time: A time vector to run the simulation

    :return: fraction_of_covergence, fraction_of_well_behavior
    """

    distances_history_start = []
    distances_history_end = []
    closest_pattern_start = []
    closest_pattern_end = []
    final_equilibrium = []
    starting_point = []

    # Run and extract data
    for i in range(N):
        nn.randomize_pattern()

        # Let's get the distances
        start = nn.o
        starting_point.append(start)

        # Calculate the closest pattern at the beginning
        append_distances_history(start, patterns, closest_pattern_start,
                                 distances_history_start)

        # Run the simulation and get the final equilibrum
        nn.run_network_simulation(time)
        end = nn.o
        final_equilibrium.append(end)

        # Calculate the closest pattern at the end
        append_distances_history(end, patterns, closest_pattern_end,
                                 distances_history_end)

    # Let;s calculate how many patterns ended up in the fix points
    tolerance = 1e-10
    fraction_of_convergence = 0
    for distances_end in distances_history_end:
        minimal_distance = min(distances_end.values())
        if minimal_distance < tolerance:
            fraction_of_convergence += 1

    fraction_of_convergence = fraction_of_convergence * 1.0 / N

    # Let's calculate how many of the patterns ended up in the one that they started closer too
    fraction_of_well_behaviour = [start - end for start, end in zip(closest_pattern_start, closest_pattern_end)].count(0)
    fraction_of_well_behaviour = fraction_of_well_behaviour * 1.0 / N
    # import IPython
    # IPython.embed()

    return fraction_of_convergence, fraction_of_well_behaviour