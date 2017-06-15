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
minicolumns = 70
n_patterns = 60

dt = 0.001

# Timings parameters
tau_z_pre = 0.150
tau_p = 5.0

# Traiming parameters
training_time = 0.100
inter_sequence_interval = 1.0
inter_pulse_interval = 0.0

n = 5
T_cue = 0.100
T_recall = 10.0
epochs = 3

patterns_indexes = [i for i in range(n_patterns)]

# Sequence protocol
number_of_sequences = 3
half_width = 2
units_to_overload = [0, 1]

# Build chain protocol
chain_protocol = Protocol()
sequences = chain_protocol.create_overload_chain(number_of_sequences, half_width, units_to_overload)

beta = False
value = 1.0
inhbition = -1.0
extension = 5
decay_factor = 0.5
sequence_decay  = 1.0

manager = create_artificial_manager(hypercolumns, minicolumns, sequences, value=value, inhibition=inhbition,
                                    extension=extension, decay_factor=decay_factor, sequence_decay=sequence_decay,
                                    dt=dt, BCPNNFast=BCPNNFast, NetworkManager=NetworkManager, ampa=True, beta=beta)

