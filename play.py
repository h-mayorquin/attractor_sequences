from __future__ import print_function

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable

from connectivity_functions import get_beta, get_w, softmax
from connectivity_functions import calculate_probability, calculate_coactivations
from data_transformer import build_ortogonal_patterns
from network import BCPNN

np.set_printoptions(suppress=True)
sns.set(font_scale=2.0)

hypercolumns = 3
minicolumns = 3

patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
patterns = list(patterns_dic.values())

P = calculate_coactivations(patterns)
p = calculate_probability(patterns)

w = get_w(P, p)
beta = get_beta(p)

# Parameters and network intitiation
tau_z_post = 0.240
tau_z_pre = 0.240
g_a = 0
tau_a = 2.7

dt = 0.01
T_simulation = 1.0
T_training = 10.0
simulation_time = np.arange(0, T_simulation + dt, dt)
training_time = np.arange(0, T_training + dt, dt)

prng = np.random.RandomState(seed=0)

#
nn = BCPNN(hypercolumns, minicolumns, beta, w, p_pre=p, p_post=p, p_co=P,
           tau_z_post=tau_z_post, tau_z_pre=tau_z_pre,
           tau_a=tau_a, g_a=g_a, M=2, prng=prng, k=0)

# This is the training
I = np.array((1, 0, 0, 0, 1, 0, 0, 0, 1))
print('Initial pattern', nn.o)
nn.run_network_simulation(simulation_time, I=I)
print('Final pattern', nn.o)


#Plotting goes here
#from plotting_functions import plot_quantity_history
#plot_quantity_history(dic_history, 'o')
