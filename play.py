from __future__ import print_function

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable

from connectivity_functions import get_beta, get_w, softmax
from connectivity_functions import calculate_probability, calculate_coactivations
from data_transformer import transform_normal_to_neural_single, transform_neural_to_normal
from data_transformer import transform_neural_to_normal_single
from network import BCPNN

np.set_printoptions(suppress=True)
sns.set(font_scale=2.0)


pattern1 = transform_normal_to_neural_single(np.array((1, 0, 0)))
pattern2 = transform_normal_to_neural_single(np.array((0, 1, 0)))
pattern3 = transform_normal_to_neural_single(np.array((0, 0, 1)))
patterns = [pattern1, pattern2]

P = calculate_coactivations(patterns)
p = calculate_probability(patterns)

w = get_w(P, p)
beta = get_beta(p)

# Parameters
tau_z_post = 0.240
tau_z_pre = 0.240
g_a = 0
tau_a = 2.7

nn = BCPNN(beta, w, p_pre=p, p_post=p, p_co=P, tau_z_post=tau_z_post, tau_z_pre=tau_z_pre,
           tau_a=tau_a, g_a=g_a, M=2)

dt = 0.01
T = 1
time = np.arange(0, T + dt, dt)

dic_history = nn.run_network_simulation(time, save=True)

# Plotting goes here
from plotting_functions import plot_quantity_history
plot_quantity_history(dic_history, 'o')