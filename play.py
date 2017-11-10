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
from analysis_functions import calculate_timings, calculate_recall_success, calculate_recall_success_sequences
from analysis_functions import calculate_excitation_inhibition_ratio, get_excitation, get_inhibition, subsequence
from analysis_functions import calculate_total_connections
from plotting_functions import plot_weight_matrix
from analysis_functions import calculate_angle_from_history, calculate_winning_pattern_from_distances
from analysis_functions import calculate_patterns_timings

from connectivity_functions import create_artificial_manager

def excitation_from_value(value, hypercolumns, minicolumns, n_patterns):
    excitation_normal = value * hypercolumns + value * (hypercolumns - 1)
    normal = ((n_patterns - 1.0) / n_patterns) * (excitation_normal)

    excitation_first =  excitation_first = value * (hypercolumns - 1)
    first = (1.0 / n_patterns) * (excitation_first)

    excitation_total = normal + first

    return excitation_total


# Patterns parameters
# Patterns parameters
hypercolumns = 4
minicolumns = 50
n_patterns = 50

# Manager properties
dt = 0.001
T_recalling = 5.0
values_to_save = ['o', 's', 'z_pre', 'z_post', 'p_pre', 'p_post', 'p_co', 'z_co', 'w']

# Protocol
training_time = 0.1
inter_sequence_interval = 0.1
inter_pulse_interval = 0.0
epochs = 1
tau_z_pre = 0.500

# Build the network
nn = BCPNNFast(hypercolumns, minicolumns, tau_z_pre=tau_z_pre)

# Build the manager
manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

# Build the protocol for training
protocol = Protocol()
patterns_indexes = [i for i in range(n_patterns)]
protocol.simple_protocol(patterns_indexes, training_time=training_time, inter_pulse_interval=inter_pulse_interval,
                         inter_sequence_interval=inter_sequence_interval, epochs=epochs)

# Train
# epoch_history = manager.run_network_protocol(protocol=protocol, verbose=True)

z_pre = manager.history['z_pre']

time = np.arange(0, training_time * n_patterns + inter_sequence_interval, dt)
time = np.arange(0, 10.0, 0.01)
y = np.exp(-time / 2.0)

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111)
ax.plot(time, y)
aux = int(2*training_time / dt)
aux = 0
ax.fill_between(time, 0, y)

fig.patch.set_visible(False)
ax.axis('off')
plt.show()
# Save the figure
# fname = './plots/filter.svg'
plt.savefig('test.svg')





