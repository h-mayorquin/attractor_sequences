import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import numpy as np
from pprint import pprint

from network import Protocol, BCPNNFast, NetworkManager
from connectivity_functions import  artificial_connectivity_matrix
from connectivity_functions import calculate_overlap_one_to_all, calculate_overlap_one_to_one
from connectivity_functions import  calculate_random_sequence, calculate_overlap_matrix
from plotting_functions import plot_artificial_sequences
from plotting_functions import plot_winning_pattern
from analysis_functions import calculate_timings, calculate_recall_success
from plotting_functions import plot_weight_matrix

from analysis_functions import calculate_recall_success_sequences
from connectivity_functions import create_artificial_manager

import pdb
# pdb.set_trace()

# Patterns parameters
hypercolumns = 4
minicolumns = 40
n_patterns = 30

dt = 0.001
