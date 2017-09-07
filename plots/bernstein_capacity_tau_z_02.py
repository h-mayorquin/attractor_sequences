import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns

from network import Protocol, BCPNNFast, NetworkManager
from analysis_functions import calculate_recall_success_sequences
from connectivity_functions import create_artificial_manager

sns.set(font_scale=3.0)

# Patterns parameters
hypercolumns = 4
minicolumns = 30
n_patterns = 10

dt = 0.001

# Manager properties
dt = 0.001
T_recall = 5.0
T_cue = 0.100
n = 10
values_to_save = ['o']

# Protocol
training_time = 0.1
inter_sequence_interval = 1.0
inter_pulse_interval = 0.0
epochs = 3

# Sequence structure
number_of_sequences = 2
half_width = 2

sigma = 0

tau_z_vector = np.arange(0.025, 0.525, 0.025)
overlaps = [2, 3, 4]
total_success_list_tau_z = []
total_success_list_tau_z_var = []
total_success_list_tau_z_min = []

for overlap in overlaps:
    print(overlap)
    total_success_tau_z = np.zeros(tau_z_vector.size)
    total_success_tau_z_var = np.zeros(tau_z_vector.size)
    total_success_tau_z_min = np.zeros(tau_z_vector.size)
    for tau_z_index, tau_z_pre in enumerate(tau_z_vector):
        # Build the network
        nn = BCPNNFast(hypercolumns, minicolumns, tau_z_pre=tau_z_pre, sigma=sigma)
        # Buidl the manager
        manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

        # Build chain protocol
        chain_protocol = Protocol()
        units_to_overload = [i for i in range(overlap)]
        sequences = chain_protocol.create_overload_chain(number_of_sequences, half_width, units_to_overload)
        chain_protocol.cross_protocol(sequences, training_time=training_time,
                                inter_sequence_interval=inter_sequence_interval, epochs=epochs)

        # Run the manager
        manager.run_network_protocol(protocol=chain_protocol, verbose=False)

        successes = calculate_recall_success_sequences(manager, T_recall=T_recall, T_cue=T_cue, n=n,
                                                       sequences=sequences)
        # Store
        total_success_tau_z[tau_z_index] = np.mean(successes)
        total_success_tau_z_min[tau_z_index] = np.min(successes)
        total_success_tau_z_var[tau_z_index] = np.var(successes)

    total_success_list_tau_z.append(total_success_tau_z)
    total_success_list_tau_z_var.append(total_success_tau_z_var)
    total_success_list_tau_z_min.append(total_success_tau_z_min)

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111)
color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
color_list = color_list[:len(overlaps)]

for overlap, total_success_tau_z, color in zip(overlaps, total_success_list_tau_z, color_list):
    ax.plot(tau_z_vector, total_success_tau_z, '*-', markersize=15, color=color,
            label='overlap = ' + str(overlap))

for overlap, total_success_tau_z, color in zip(overlaps, total_success_list_tau_z_min, color_list):
    ax.plot(tau_z_vector, total_success_tau_z, markersize=15, color=color, linestyle='--')

ax.axhline(0, color='black', alpha=0.2)
ax.set_ylim([-5, 105])

ax.set_xlabel(r'$\tau_z$  NMDA')
ax.set_ylabel('Success')

ax.legend();

fname = './plots/capacity_tau_z.pdf'
plt.savefig(fname, format='pdf', dpi=100, bbox_inches='tight', frameon=True, transparent=False)