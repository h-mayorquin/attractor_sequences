from __future__ import print_function

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable

from connectivity_functions import get_beta, get_w, softmax
from connectivity_functions import calculate_probability, calculate_coactivations
from data_transformer import transform_normal_to_neural_single, transform_neural_to_normal
from data_transformer import transform_neural_to_normal_single, transform_singleton_to_normal
from data_transformer import build_ortogonal_patterns
from network import BCPNN

np.set_printoptions(suppress=True)
sns.set(font_scale=2.0)

hypercolumns = 3
minicolumns = 3

patterns_dic =
patterns = [pattern1, pattern2, pattern3]

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
T = 1
time = np.arange(0, T + dt, dt)

distances_history = []
closest_pattern = []
final_equilibrium = []
starting_point = []

N = 0

# Run and extract data
for i in range(N):
    nn = BCPNN(hypercolumns, minicolumns, beta, w, p_pre=p, p_post=p, p_co=P,
               tau_z_post=tau_z_post, tau_z_pre=tau_z_pre,
               tau_a=tau_a, g_a=g_a, M=2)

    # Let's get the distances
    start = nn.o
    starting_point.append(start)
    distances = {}
    aux = [np.linalg.norm(start - pattern) for pattern in patterns]
    distances_history.append({k:v for k,v in enumerate(aux)})
    closest_pattern.append(aux.index(min(aux)))

    # Run the simulation and get the final equilibrum
    dic_history = nn.run_network_simulation(time, save=True)
    final_equilibrium.append(transform_neural_to_normal_single(nn.o, minicolumns))



# Plotting goes here
#from plotting_functions import plot_quantity_history
#plot_quantity_history(dic_history, 'o')\