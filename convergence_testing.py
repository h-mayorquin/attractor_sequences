from __future__ import print_function

import numpy as np

from connectivity_functions import get_beta, get_w
from connectivity_functions import calculate_probability, calculate_coactivations
from data_transformer import build_ortogonal_patterns
from convergence_functions import test_convergence_ratios
from network import BCPNN

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set(font_scale=3.0)

np.set_printoptions(suppress=True)

dt = 0.01
T = 1
time = np.arange(0, T + dt, dt)


g_a = 0.0  # No adaptation
g_beta = 1.0  # No bias gain
g_w = 1.0  # No weight gain
prng = np.random.RandomState(seed=0)
N = 10

convergence_fractions = []
well_behaved_fractions = []

numbers = np.arange(2, 11, dtype=int)

for number in numbers:

    hypercolumns = number
    minicolumns = number

    patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
    patterns = list(patterns_dic.values())

    P = calculate_coactivations(patterns)
    p = calculate_probability(patterns)

    w = get_w(P, p)
    beta = get_beta(p)

    nn = BCPNN(hypercolumns, minicolumns, beta, w, p_pre=p, p_post=p, p_co=P,
               g_a=g_a, g_beta=g_beta, g_w=g_w, prng=prng)

    fraction_of_convergence, fraction_of_well_behaved = test_convergence_ratios(nn, N, time, patterns)

    convergence_fractions.append(fraction_of_convergence * 100)
    well_behaved_fractions.append(fraction_of_well_behaved * 100)


fig = plt.figure(figsize=(16 ,12))

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(numbers, convergence_fractions, '*-', markersize=15)
ax1.set_xlabel('# Hypercolumns and Minicolumns')
ax1.set_ylabel('% of success')

ax1.set_ylim([0, 105])
ax1.set_xlim((0, numbers[-1] + 1))
ax1.set_title('Convergence Percentange')


ax2.plot(numbers, well_behaved_fractions, '*-', markersize=15)
ax2.set_xlabel('# Hypercolumns and Minicolumns')
ax2.set_ylabel('% of success')


ax2.set_ylim([0, 105])
ax2.set_xlim((0, numbers[-1] + 1))
ax2.set_title('Well behaved percentange')

plt.show()