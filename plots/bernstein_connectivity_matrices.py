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


normal_plot = False
overlaped_plot = False
overloaded_pot = True

def set_text(ax, coordinate_from, coordinate_to, fontsize=25):
    message = str(coordinate_from) + '->' + str(coordinate_to)
    ax.text(coordinate_from, coordinate_to, message, ha='center', va='center', rotation=315, fontsize=fontsize)

sigma = 0

if normal_plot:
    # Patterns parameters
    hypercolumns = 4
    minicolumns = 10
    n_patterns = 10

    dt = 0.001

    # Manager properties
    dt = 0.001
    T_recalling = 5.0
    values_to_save = ['o']

    # Protocol
    training_time = 0.1
    inter_sequence_interval = 3.0
    inter_pulse_interval = 0.0
    epochs = 3

    # Build the network
    nn = BCPNNFast(hypercolumns, minicolumns, sigma=0)

    # Build the manager
    manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

    # Build the protocol
    # Build the protocol for training
    protocol = Protocol()
    patterns_indexes = [i for i in range(n_patterns)]
    protocol.simple_protocol(patterns_indexes, training_time=training_time, inter_pulse_interval=inter_pulse_interval,
                             inter_sequence_interval=inter_sequence_interval, epochs=epochs)

    # Train
    epoch_history = manager.run_network_protocol(protocol=protocol, verbose=True)

    # Now plotting
    w = nn.w

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)

    w = w[:nn.minicolumns, :nn.minicolumns]

    aux_max = np.max(np.abs(w))

    cmap = 'coolwarm'
    im = ax.imshow(w, cmap=cmap, interpolation='None', vmin=-aux_max, vmax=aux_max)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax.get_figure().colorbar(im, ax=ax, cax=cax)

    # Add text numbers
    for i in range(n_patterns - 1):
        coordinate_from = i
        coordinate_to = i + 1
        set_text(ax, coordinate_from, coordinate_to, fontsize=25)


    # Editing
    ax.grid()
    ax.set_title('Simple sequence')
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')

    fname = './plots/matrix_normal.pdf'
    plt.savefig(fname, format='pdf', dpi=100, bbox_inches='tight', frameon=True, transparent=False)

    plt.show()

if overlaped_plot:
    # Patterns parameters
    hypercolumns = 4
    minicolumns = 15
    n_patterns = 10

    dt = 0.001

    # Manager properties
    dt = 0.001
    T_recalling = 5.0
    values_to_save = ['o']

    # Protocol
    training_time = 0.3
    inter_sequence_interval = 1.0
    inter_pulse_interval = 0.0
    epochs = 3

    # Build the network
    nn = BCPNNFast(hypercolumns, minicolumns, sigma=0)

    # Build the manager
    manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

    # Build chain protocol

    chain_protocol = Protocol()
    sequences = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 2, 3, 4, 9, 10]]
    chain_protocol.cross_protocol(sequences, training_time=training_time,
                                  inter_sequence_interval=inter_sequence_interval, epochs=epochs)
    print(sequences)

    # Train
    manager.run_network_protocol(protocol=chain_protocol, verbose=True)

    # Now plotting
    w = nn.w

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)

    w = w[:nn.minicolumns, :nn.minicolumns]

    aux_max = np.max(np.abs(w))

    cmap = 'coolwarm'
    im = ax.imshow(w, cmap=cmap, interpolation='None', vmin=-aux_max, vmax=aux_max)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax.get_figure().colorbar(im, ax=ax, cax=cax)

    # Add text
    fontsize = 18
    for sequence in sequences:
        print(sequence)
        for i in range(len(sequence) - 1):
            coordinate_from = sequence[i]
            coordinate_to = sequence[i + 1]
            set_text(ax, coordinate_from=coordinate_from, coordinate_to=coordinate_to, fontsize=fontsize)


    # Editing
    ax.grid()
    ax.set_title('Sequences with overlap')
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')

    fname = './plots/matrix_overlap.pdf'
    plt.savefig(fname, format='pdf', dpi=100, bbox_inches='tight', frameon=True, transparent=False)

    plt.show()

if overloaded_pot:
    # Patterns parameters
    hypercolumns = 4
    minicolumns = 23
    n_patterns = 10

    dt = 0.001

    # Manager properties
    dt = 0.001
    T_recalling = 5.0
    values_to_save = ['o']

    # Protocol
    training_time = 0.3
    inter_sequence_interval = 3.0
    inter_pulse_interval = 0.0
    epochs = 3

    # Build the network
    nn = BCPNNFast(hypercolumns, minicolumns, sigma=0)
    # Build the manager
    manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

    # Build chain protocol

    chain_protocol = Protocol()
    sequences = [[1, 2, 0, 3, 4], [5, 6, 0, 7, 8], [9, 10, 0, 11, 12], [13, 14, 0, 15, 16]]
    chain_protocol.cross_protocol(sequences, training_time=training_time,
                                  inter_sequence_interval=inter_sequence_interval, epochs=epochs)
    print(sequences)

    # Train
    manager.run_network_protocol(protocol=chain_protocol, verbose=True)

    # Now plotting
    w = nn.w

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)

    w = w[:nn.minicolumns, :nn.minicolumns]

    aux_max = np.max(np.abs(w))

    cmap = 'coolwarm'
    im = ax.imshow(w, cmap=cmap, interpolation='None', vmin=-aux_max, vmax=aux_max)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax.get_figure().colorbar(im, ax=ax, cax=cax)

    # Add text
    fontsize = 8
    for sequence in sequences:
        print(sequence)
        for i in range(len(sequence) - 1):
            coordinate_from = sequence[i]
            coordinate_to = sequence[i + 1]
            set_text(ax, coordinate_from=coordinate_from, coordinate_to=coordinate_to, fontsize=fontsize)

    # Editing
    ax.grid()
    ax.set_title('High overload scenario')
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')

    fname = './plots/matrix_overload.pdf'
    plt.savefig(fname, format='pdf', dpi=100, bbox_inches='tight', frameon=True, transparent=False)

    plt.show()
