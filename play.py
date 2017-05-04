import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pprint import pprint

from network import Protocol, BCPNNFast, NetworkManager
from plotting_functions import plot_winning_pattern
from analysis_functions import calculate_recall_success, calculate_timings
from analysis_functions import create_artificial_matrix
from plotting_functions import plot_weight_matrix, plot_network_activity_angle
from analysis_functions import calculate_angle_from_history, calculate_winning_pattern_from_distances

# Patterns parameters
hypercolumns = 4
minicolumns = 60
n_patterns = 30

# Manager properties
dt = 0.001
T_recalling = 5.0
values_to_save = ['o', 'p_pre', 'p_post', 'p_co', 'w']

# Protocol
training_time = 0.1
inter_sequence_interval = 1.0
inter_pulse_interval = 0.0
epochs = 3

# Build the network
nn = BCPNNFast(hypercolumns, minicolumns)
# Build the manager
manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

# Build the protocol for training
protocol = Protocol()
patterns_indexes = [i for i in range(n_patterns)]
protocol.simple_protocol(patterns_indexes, training_time=training_time, inter_pulse_interval=inter_pulse_interval,
                         inter_sequence_interval=inter_sequence_interval, epochs=epochs)


# Train
manager.run_network_protocol(protocol=protocol, verbose=True)

# Extract history
w = manager.history['w']
p_co = manager.history['p_co']
o = manager.history['o']
p = manager.history['p_pre']

w_indexes = [(3 , 2), (4, 3), (6, 5)]
p_co_indexes = [(3 , 2), (4, 3), (6, 5)]
o_indexes = [2, 3, 5, 6]
p_indexes = [2, 3, 5, 6]

# Plot
time = np.arange(0, manager.T_total, dt)
time = np.linspace(0, manager.T_total, o.shape[0])
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2)

ax = fig.add_subplot(gs[1, 0])
for w_index in w_indexes:
    ax.plot(time, w[:, w_index[0], w_index[1]], label=str(w_index))
    ax.legend()

ax = fig.add_subplot(gs[0, 0])
for o_index in o_indexes:
    ax.plot(time, o[:, o_index], label=str(o_index))
    ax.legend()

# P_co
ax = fig.add_subplot(gs[0, 1])
for p_co_index in p_co_indexes:
    ax.plot(time, p_co[:, p_co_index[0], p_co_index[1]], label=str(p_co_index))
    ax.legend()

ax = fig.add_subplot(gs[1, 1])
for p_index in p_indexes:
    ax.plot(time, p[:, p_index], label=str(p_index))
    ax.legend()

plt.show()

