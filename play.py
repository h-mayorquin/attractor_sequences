from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from connectivity_functions import get_beta, get_w
from connectivity_functions import calculate_probability, calculate_coactivations
from data_transformer import build_ortogonal_patterns
from network import BCPNN

np.set_printoptions(suppress=True)

hypercolumns = 3
minicolumns = 3

patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
patterns = list(patterns_dic.values())

P = calculate_coactivations(patterns)
p = calculate_probability(patterns)

w = get_w(P, p)
beta = get_beta(p)

dt = 0.01
T_simulation = 10.0
T_training = 1.0
simulation_time = np.arange(0, T_simulation + dt, dt)
training_time = np.arange(0, T_simulation + dt, dt)

prng = np.random.RandomState(seed=0)

tolerance = 1e-5

g_a_set = np.arange(0, 110, 10)
g_beta_set = np.arange(0, 22, 2)
g_w_set = np.arange(0, 12, 2)

nn = BCPNN(hypercolumns, minicolumns, beta, w, g_a=1, g_beta=1.0, g_w=1.0, g_I=10.0, prng=prng)
nn.randomize_pattern()

if False:
    final_states = []
    for i in range(10):
        nn.reset_values()
        print(nn.o)
        nn.run_network_simulation(training_time)
        print(nn.o)
        final_states.append(nn.o)


if False:
    I = patterns[0]

    for i in range(20):
        nn.randomize_pattern()
        nn.run_network_simulation(training_time, I=I)
        print('I', I)
        print('o', nn.o)
        print('Difference', I - nn.o)
        print('Norm', np.linalg.norm(I - nn.o))

from convergence_functions import calculate_convergence_ratios

# x, y = calculate_convergence_ratios(nn, 10, training_time, patterns)

if True:
    for pattern in reversed(patterns):
        I = pattern
        nn.randomize_pattern()
        print('---------')
        print('I')
        print(I)
        # Run the network for one minut clamped to the patternr
        nn.k = 1.0
        nn.run_network_simulation(training_time, I=I)
        # Run the network free
        print('---------')
        print('This should look like I')
        print(nn.o)
        nn.k = 0.0
        nn.run_network_simulation(training_time)
        print('---------')
        print('---------')
        print(nn.o)



history = nn.run_network_simulation(simulation_time, save=True)
o = history['o']
s = history['s']

plt.imshow(o, aspect='auto', interpolation='None')
plt.colorbar()
plt.show()