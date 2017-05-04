import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from network import NetworkManager, BCPNNFast, Protocol

# Patterns parameters
hypercolumns = 4
minicolumns = 15
n_patterns = 10

# Manager properties
dt = 0.001
T_recalling = 5.0
values_to_save = ['o']


# Protocol
training_time = 0.1
inter_sequence_interval = 1.0
inter_pulse_interval = 0.0
epochs = 30

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
epoch_history = manager.run_network_protocol(protocol=protocol, verbose=False, values_to_save_epoch=['w'])
w_history = epoch_history['w']

from_pattern = 0
w_epoch = [w_t[:, from_pattern].reshape(nn.hypercolumns, nn.minicolumns) for w_t in w_history]
w_epoch = [np.mean(w, axis=0) for w in w_epoch]
w_epoch = np.array(w_epoch)

if True:
    cmap_string = 'nipy_spectral'
    cmap_string = 'hsv'
    cmap_string = 'Paired'

    cmap = matplotlib.cm.get_cmap(cmap_string)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=minicolumns)

    fig = plt.figure(figsize=(16, 12))
    import seaborn as sns
    sns.set(font_scale=2.0)

    ax = fig.add_subplot(111)

    for index, w in enumerate(w_epoch.T):
        ax.plot(w, '*-', color=cmap(norm(index)), linewidth=3, markersize=14, label=str(index))

    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlim([-1, epochs + 2])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Connectivity weight')
    ax.set_title('AMPA connectivity after training going from 0 to different attractors')
    ax.legend()

    fig.savefig('stability.png', dpi=200)
    plt.close()

if True:
    from_pattern = 4
    w_final = w_history[-1][:, from_pattern].reshape((hypercolumns, minicolumns)).mean(axis=0)

    fig = plt.figure(figsize=(16, 12))
    import seaborn as sns
    sns.set(font_scale=2.0)

    ax = fig.add_subplot(111)
    ax.plot(w_final, '*-', linewidth=3, markersize=13)
    ax.axhline(y=0, color='grey', linestyle='--')

    ax.set_xlabel('Attractor')
    ax.set_ylabel('Weight')
    ax.set_title('Final connections emanating from attractor ' + str(from_pattern))

    fig.savefig('conn_from.png', dpi=200)
    plt.close()
