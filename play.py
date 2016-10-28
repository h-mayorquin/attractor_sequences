from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from network import BCPNN
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

nn = BCPNN(hypercolumns, minicolumns)
nn.k = 1.0
nn.randomize_pattern()

dt = 0.001
T_training = 1.0
T_simulation = 20.0
training_time = np.arange(0, T_training + dt, dt)
simulation_time = np.arange(0, T_simulation + dt, dt)

for pattern in patterns:
    history = nn.run_network_simulation(training_time, I=pattern, save=True)

a = history['a']
a_first = a[:, 0:minicolumns]
time_aux = np.arange(0, n_patterns * (T_training + dt), dt)

fig = plt.figure()
ax1 = fig.add_subplot(121)

for unit, a_unit in enumerate(a_first.T):
    ax1.plot(time_aux, a_unit, label=str(unit))

ax1.legend()


# After
nn.empty_history()
history = nn.run_network_simulation(simulation_time, save=True)

a = history['a']
a_first = a[:, 0:minicolumns]
time_aux = np.arange(0, T_simulation + dt, dt)

ax2 = fig.add_subplot(122)

for unit, a_unit in enumerate(a_first.T):
    ax2.plot(time_aux, a_unit, label=str(unit))

ax2.legend()
plt.show()
