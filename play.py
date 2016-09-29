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

prng = np.random.RandomState(seed=0)


def calculate_closest_pattern(distance_to_patterns_list):

    return distance_to_patterns_list.index(min(distance_to_patterns_list))


def calculate_distances_to_fix_points(point, patterns):

    return [np.linalg.norm(point - pattern) for pattern in patterns]


# Run and extract data
for i in range(N):
    nn = BCPNN(hypercolumns, minicolumns, beta, w, p_pre=p, p_post=p, p_co=P,
               tau_z_post=tau_z_post, tau_z_pre=tau_z_pre,
               tau_a=tau_a, g_a=g_a, M=2, prng=prng)

    # Let's get the distances
    start = nn.o
    starting_point.append(start)

    # Calculate the closes pattern at the beginning
    aux = calculate_distances_to_fix_points(start, patterns)
    distances_history_start.append({k: v for k, v in enumerate(aux)})
    closest_pattern_start.append(min(distances_history_start[-1], key=distances_history_start[-1].get))

    # Store the distance at the beginning



    # Run the simulation and get the final equilibrum
    dic_history = nn.run_network_simulation(time, save=False)
    end = nn.o
    final_equilibrium.append(end)

    # Calculate the closes pattern at the end
    aux = calculate_distances_to_fix_points(end, patterns)
    closest_pattern_end.append(calculate_closest_pattern(aux))
    distances_history_end.append({k: v for k, v in enumerate(aux)})

print(closest_pattern_end)
print(closest_pattern_start)
#Plotting goes here
#from plotting_functions import plot_quantity_history
#plot_quantity_history(dic_history, 'o')
