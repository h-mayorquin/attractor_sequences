from __future__ import print_function
import numpy as np


def calculate_closest_pattern_list(distance_to_patterns_list):

    return distance_to_patterns_list.index(min(distance_to_patterns_list))


def calculate_distances_to_fix_points_list(point, patterns):

    return [np.linalg.norm(point - pattern) for pattern in patterns]

def calculate_closest_pattern_dictionary(distances_dictionary):

    return min(distances_dictionary, key=distances_dictionary.get)

def calculate_distances_to_fix_points_dictionary(point, patterns):

    return {pattern_number: np.linalg.norm(point - pattern) for pattern_number, pattern in enumerate(patterns)}



