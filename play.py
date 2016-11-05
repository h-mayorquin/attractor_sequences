from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from network import BCPNN, NetworkManager
from data_transformer import build_ortogonal_patterns
from connectivity_functions import calculate_coactivations, calculate_probability, get_w, get_beta
from analysis_functions import calculate_angle_from_history
from analysis_functions import calculate_winning_pattern_from_distances, calculate_patterns_timings

np.set_printoptions(suppress=True)

hypercolumns = 10
minicolumns = 10
n_patterns = 10  # Number of patterns

patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
patterns = list(patterns_dic.values())
patterns = patterns[:n_patterns]

# Build the network
tau_z_pre = 0.250
tau_z_post = 0.500

nn = BCPNN(hypercolumns, minicolumns, tau_z_post=tau_z_post, tau_z_pre=tau_z_pre)
# nn.tau_p = 1.0


dt = 0.001
T_training = 1.0
time_training = np.arange(0, T_training + dt, dt)
T_ground = 1.0
time_ground = np.arange(0, T_ground + dt, dt)
values_to_save = ['o', 'z_pre', 'z_post', 'p_pre', 'p_post', 'p_co', 'z_co', 'w']

manager = NetworkManager(nn=nn, values_to_save=values_to_save)

repetitions = 2
resting_state = True

for i in range(repetitions):
    print('repetitions', i)
    for pattern in patterns:
        nn.k = 1.0
        manager.run_network(time=time_training, I=pattern)
        nn.k = 0.0
        if resting_state:
            manager.run_network(time=time_ground)

history = manager.history

if resting_state:
    T_total = n_patterns * repetitions * (T_training + T_ground + dt + dt)
else:
    T_total = n_patterns * repetitions * (T_training + dt)

total_time = np.arange(0, T_total , dt)

z_pre_hypercolum = history['z_pre'][..., :minicolumns]
z_post_hypercolum = history['z_post'][..., :minicolumns]
o_hypercolum = history['o'][..., :minicolumns]
p_pre_hypercolum = history['p_pre'][..., :minicolumns]
p_post_hypercolum = history['p_post'][..., :minicolumns]

p_co = history['p_co']
z_co = history['z_co']
w = history['w']

p_co01 = p_co[:, 0, 1]
p_co10 = p_co[:, 1, 0]

z_co01 = z_co[:, 0, 1]
z_co10 = z_co[:, 1, 0]

w01 = w[:, 0, 1]
w10 = w[:, 1, 0]

aux01 = p_co01 / (p_pre_hypercolum[:, 0] * p_post_hypercolum[:, 1])
aux10 = p_co10 / (p_pre_hypercolum[:, 1] * p_post_hypercolum[:, 0])

import seaborn as sns

# Plot the traces
fig = plt.figure(figsize=(16, 12))
ax11 = fig.add_subplot(421)
ax12 = fig.add_subplot(422)
ax21 = fig.add_subplot(423)
ax22 = fig.add_subplot(424)
ax31 = fig.add_subplot(425)
ax32 = fig.add_subplot(426)
ax41 = fig.add_subplot(427)
ax42 = fig.add_subplot(428)

fig.tight_layout()
# fig.subplots_adjust(right=0.8)

for index in range(minicolumns):
    # Plot the activities
    ax11.plot(total_time, o_hypercolum[:, index], label=str(index))
    ax12.plot(total_time, o_hypercolum[:, index], label=str(index))

for index in range(n_patterns):
    # Plot the z post and pre traces in the same graph
    ax21.plot(total_time, z_pre_hypercolum[:, index], label='pre ' + str(index))
    ax21.plot(total_time, z_post_hypercolum[:, index], '--', label='post ' + str(index))

    # Plot the pre and post probabilties in the same graph
    ax22.plot(total_time, p_pre_hypercolum[:, index], label='pre ' + str(index))
    ax22.plot(total_time, p_post_hypercolum[:, index], '--', label='post ' + str(index))

# Plot z_co and p_co in the same graph
ax31.plot(total_time, z_co01, label='zco_01')
ax31.plot(total_time, z_co10, label='zco_10')

# Plot the aux quantity
ax32.plot(total_time, aux01, label='aux01')
ax32.plot(total_time, aux10, label='aux10')

ax41.plot(total_time, p_co01, '-', label='pco_01')
ax41.plot(total_time, p_co10, '-',label='pco_10')

ax42.plot(total_time, w01, label='01')
ax42.plot(total_time, w10, label='10')

axes = fig.get_axes()
for ax in axes:
    ax.set_xlim([0, T_total])
    ax.legend()

ax11.set_ylim([-0.1, 1.1])
ax12.set_ylim([-0.1, 1.1])

ax21.set_title('z-traces')
ax22.set_title('probabilities')
ax31.set_title('z_co')
ax32.set_title('Aux')
ax41.set_title('p_co')
ax42.set_title('w')

plt.show()

