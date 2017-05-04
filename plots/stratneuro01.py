import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from network import NetworkManager, BCPNNFast, Protocol
from plotting_functions import plot_weight_matrix, plot_winning_pattern

# Patterns parameters
hypercolumns = 4
minicolumns = 30
n_patterns = 15

# Manager properties
dt = 0.001
T_recalling = 5.0
values_to_save = ['o']


# Protocol
training_time = 0.1
inter_sequence_interval = 2.0
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
epoch_history = manager.run_network_protocol(protocol=protocol, verbose=True, values_to_save_epoch=['w'])
if True:
    fig = plt.figure(figsize=(16, 12))
    sns.set(font_scale=2.5)
    sns.set_style("whitegrid", {'axes.grid': False})
    ax = fig.add_subplot(111)
    plot_weight_matrix(nn, ampa=False, one_hypercolum=True, ax=ax)
    fig.savefig('matrix.png', dpi=200)
    plt.close()

# Recall
if False:
    T_recall = 5.0
    T_cue = training_time
    manager.run_network_recall(T_recall=T_recall, T_cue=T_cue, I_cue=0)

    fig = plt.figure(figsize=(16, 12))
    sns.set(font_scale=2.0)
    sns.set_style("whitegrid", {'axes.grid': False})
    ax = fig.add_subplot(111)
    plot_winning_pattern(manager, ax=ax, remove=0.030)
    fig.savefig('winning.png', dpi=200)
    plt.close()
