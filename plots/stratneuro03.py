import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from network import NetworkManager, BCPNNFast, Protocol
from analysis_functions import calculate_recall_success, calculate_timings

# Patterns parameters
hypercolumns = 4
minicolumns_range = np.arange(40, 80, 10)

# Manager properties
dt = 0.001
T_recall = 3.0
values_to_save = ['o']

# Protocol
training_time = 0.1
inter_sequence_interval = 1.0
inter_pulse_interval = 0.0
epochs = 3

# Build chain protocol
number_of_sequences = 5
half_width = 2
units_to_overload = [0, 1]

chain_protocol = Protocol()
chain = chain_protocol.create_overload_chain(number_of_sequences, half_width, units_to_overload)
chain_protocol.cross_protocol(chain, training_time=training_time, inter_sequence_interval=inter_sequence_interval,
                        epochs=epochs)

# Calculate
n = 10
success_array = np.zeros((len(chain), minicolumns_range.size))
for minicolumn_index, minicolumns in enumerate(minicolumns_range):
    print(minicolumns)
    # Build the network
    nn = BCPNNFast(hypercolumns, minicolumns)

    # Build the manager
    manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

    # Train
    manager.run_network_protocol(chain_protocol, verbose=False, values_to_save_epoch=None, reset=True, empty_history=True)

    # Test recall
    for sequence_index, sequence in enumerate(chain):
        success = calculate_recall_success(manager, T_recall=T_recall, I_cue=sequence[0], T_cue=0.1, n=n,
                                        patterns_indexes=sequence)
        success_array[sequence_index, minicolumn_index] = success


# Plot
sns.set(font_scale=2.0)

fig = plt.figure(figsize=(16, 12))
sns.set(font_scale=2.2)
ax = fig.add_subplot(111)

ax.plot(minicolumns_range, np.mean(success_array, axis=0), '*-', markersize=14, linewidth=3, label='Mean Recall Success')

for sequence_index in range(len(chain)):
    ax.plot(minicolumns_range, success_array[sequence_index, :], '*-', markersize=10, linewidth=1, label=str(sequence_index))

ax.legend()
ax.set_ylim([-5, 105])
ax.set_xlabel('Minicolumns')
ax.set_ylabel('Success')
ax.set_title('Recall success for different subsequences')

fig.savefig('complex_recall.png', dpi=200)
plt.close()
