import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns

from network import Protocol, BCPNNFast, NetworkManager
from connectivity_functions import create_artificial_manager

sns.set(font_scale=2.0)

# Patterns parameters
hypercolumns = 4
minicolumns = 15
n_patterns = 15

# Manager properties
dt = 0.001
T_recalling = 5.0
values_to_save = []

# Protocol
training_time = 0.1
inter_sequence_interval = 1.0
inter_pulse_interval = 0.0
epochs = 2

tau_z = 0.150

# Build the network
nn = BCPNNFast(hypercolumns, minicolumns, tau_z)

# Build the manager
manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

# Build the protocol for training
protocol = Protocol()
patterns_indexes = [i for i in range(n_patterns)]
protocol.simple_protocol(patterns_indexes, training_time=training_time, inter_pulse_interval=inter_pulse_interval,
                         inter_sequence_interval=inter_sequence_interval, epochs=epochs)

# Train
manager.run_network_protocol(protocol=protocol, verbose=True)

# Artificial matrix
beta = False
value = 1.0
inhibition = -0.3
extension = 3
decay_factor = 0.45
sequence_decay = 0.0
ampa = True
self_influence = False

sequences = [[i for i in range(n_patterns)]]

manager_art = create_artificial_manager(hypercolumns, minicolumns, sequences, value, inhibition, extension, decay_factor,
                                    sequence_decay, dt, BCPNNFast, NetworkManager, ampa, beta, beta_decay=False,
                                    self_influence=self_influence)


cmap = 'coolwarm'
w = manager.nn.w
w = w[:nn.minicolumns, :nn.minicolumns]
aux_max = np.max(np.abs(w))

fig = plt.figure(figsize=(16, 12))

ax1 = fig.add_subplot(121)
im1 = ax1.imshow(w, cmap=cmap, interpolation='None', vmin=-aux_max, vmax=aux_max)
ax1.set_title('Training Procedure')

ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)
ax1.grid()

divider = make_axes_locatable(ax1)
cax1 = divider.append_axes('right', size='5%', pad=0.05)
ax1.get_figure().colorbar(im1, ax=ax1, cax=cax1)

w_art = manager_art.nn.w
w_art = w_art[:nn.minicolumns, :nn.minicolumns]
aux_max = np.max(np.abs(w))

ax2 = fig.add_subplot(122)

im2 = ax2.imshow(w_art, cmap=cmap, interpolation='None', vmin=-aux_max, vmax=aux_max)
ax2.set_title('Artificial Matrix')

ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)
ax2.grid()

divider = make_axes_locatable(ax2)
cax2 = divider.append_axes('right', size='5%', pad=0.05)
ax2.get_figure().colorbar(im2, ax=ax2, cax=cax2)

# Save the figure
fname = './plots/comparison.pdf'
plt.savefig(fname, format='pdf', dpi=90, bbox_inches='tight', frameon=False, transparent=True)
