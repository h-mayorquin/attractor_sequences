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
T = 1
time = np.arange(0, T + dt, dt)

distances_history_start = []
distances_history_end = []
closest_pattern_start = []
closest_pattern_end = []
final_equilibrium = []
starting_point = []

N = 3

# Run and extract data
for i in range(N):
    nn = BCPNN(hypercolumns, minicolumns, beta, w, p_pre=p, p_post=p, p_co=P,
               tau_z_post=tau_z_post, tau_z_pre=tau_z_pre,
               tau_a=tau_a, g_a=g_a, M=2)

    # Let's get the distances
    start = nn.o
    starting_point.append(start)

    # Calcualte the closes pattern at the beggining
    aux = [np.linalg.norm(start - pattern) for pattern in patterns]
    closest_pattern_start.append(aux.index(min(aux)))

    # Store the distance at the beggining
    distances_history_start.append({k: v for k, v in enumerate(aux)})


    # Run the simulation and get the final equilibrum
    dic_history = nn.run_network_simulation(time, save=False)
    end = nn.o
    final_equilibrium.append(end)

    # Calculate the closes pattern at the end
    aux = [np.linalg.norm(end - pattern) for pattern in patterns]
    closest_pattern_end.append(aux.index(min(aux)))
    distances_history_end.append({k: v for k, v in enumerate(aux)})

    if closest_pattern_end[-1] == closest_pattern_start[-1]:








# Plotting goes here
#from plotting_functions import plot_quantity_history
#plot_quantity_history(dic_history, 'o')\