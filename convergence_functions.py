from __future__ import print_function
import numpy as np


def calculate_arg_min_list(list):
    """
    Calculates the argument of the minimiun in a list
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
