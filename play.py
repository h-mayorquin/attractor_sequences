from __future__ import print_function

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable

from connectivity_functions import get_beta, get_w
from connectivity_functions import calculate_probability, calculate_coactivations
from data_transformer import build_ortogonal_patterns
from network import BCPNN

np.set_printoptions(suppress=True)
sns.set(font_scale=2.0)

hypercolumns = 4
minicolumns = 4

patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
patterns = list(patterns_dic.values())

P = calculate_coactivations(patterns)
p = calculate_probability(patterns)

w = get_w(P, p)
beta = get_beta(p)

dt = 0.01
T_simulation = 1.0
simulation_time = np.arange(0, T_simulation + dt, dt)

prng = np.random.RandomState(seed=0)


tolerance = 1e-5

g_a_set = np.arange(0, 110, 10)
g_beta_set = np.arange(0, 22, 2)
g_w_set = np.arange(0, 12, 2)

# Pattern to clamp
I = np.array((1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))


# Test the error
for index_1, g_a in enumerate(g_a_set):
    for index_2, g_beta in enumerate(g_beta_set):
        for index_3, g_w in enumerate(g_w_set):
            nn = BCPNN(hypercolumns, minicolumns, beta, w, p_pre=p, p_post=p, p_co=P,
                       g_a=g_a, g_beta=g_beta, g_w=g_w, prng=prng, k=0)

        nn.randomize_pattern()

        # This is the training
        nn.run_network_simulation(simulation_time, I=I)
        final = nn.o
        point_error = np.sum(I - final)
        if point_error is np.nan:
            print(g_a, g_beta, g_w)
