import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

from network import Protocol, BCPNNFast, NetworkManager
from plotting_functions import plot_winning_pattern
from analysis_functions import calculate_recall_success, calculate_timings
from analysis_functions import create_artificial_matrix
from plotting_functions import plot_weight_matrix, plot_network_activity_angle
from analysis_functions import calculate_angle_from_history, calculate_winning_pattern_from_distances


# Network parameters
minicolumns = 10
hypercolumns = 4
number_of_patterns = 5
patterns_indexes = [i for i in range(number_of_patterns)]

