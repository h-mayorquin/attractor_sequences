import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns

from network import Protocol, BCPNNFast, NetworkManager
from analysis_functions import calculate_recall_success_sequences
from connectivity_functions import create_artificial_manager

sns.set(font_scale=2.5)

# Patterns parameters
hypercolumns = 4
minicolumns = 50

dt = 0.001

# Recall
n = 10
T_cue = 0.100
T_recall = 10.0

# Artificial matrix
beta = False
value = 3
inhibition = -1
extension = 4
decay_factor = 0.3
sequence_decay = 0.0
tau_z_pre = 0.150

# Sequence structure
overlap = 2
number_of_sequences = 2
half_width = 3
extension_vector = np.arange(1, 7, 1)

total_success_list_extension = []
overloads = [2, 4, 6]
for number_of_sequences in overloads:
    total_success_extension = np.zeros(extension_vector.size)
    print('overloads', number_of_sequences)
    for extension_index, extension in enumerate(extension_vector):

        # Build chain protocol
        chain_protocol = Protocol()
        units_to_overload = [i for i in range(overlap)]
        sequences = chain_protocol.create_overload_chain(number_of_sequences, half_width, units_to_overload)

        manager = create_artificial_manager(hypercolumns, minicolumns, sequences, value=value,
                                            inhibition=inhibition,
                                            extension=extension, decay_factor=decay_factor,
                                            sequence_decay=sequence_decay,
                                            dt=dt, BCPNNFast=BCPNNFast, NetworkManager=NetworkManager, ampa=True,
                                            beta=beta)

        manager.nn.tau_z_pre = tau_z_pre

        successes = calculate_recall_success_sequences(manager, T_recall=T_recall, T_cue=T_cue, n=n,
                                                       sequences=sequences)
        total_success_extension[extension_index] = np.min(successes)
    total_success_list_extension.append(total_success_extension)

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111)
for overload, total_success_extension in zip(overloads, total_success_list_extension):
    ax.plot(extension_vector, total_success_extension, '*-', markersize=15, label='overload = ' + str(overload))

ax.axhline(0, color='black')
ax.set_ylim([-5, 115])

ax.set_xlabel('Extension')
ax.set_ylabel('Success')

ax.legend()

# Save the figure
fname = './plots/capacity_extension_overload.pdf'
plt.savefig(fname, format='pdf', dpi=90, bbox_inches='tight', frameon=True, transparent=False)


