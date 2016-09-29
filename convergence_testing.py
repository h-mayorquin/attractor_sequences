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

from convergence_functions import calculate_distances_to_fix_points_dictionary, calculate_closest_pattern_dictionary
from convergence_functions import calculate_distances_to_fix_points_list, calculate_closest_pattern_list

def save_distances_history(point, patterns, save_dictionary=True):




# Run and extract data
for i in range(N):
    nn = BCPNN(hypercolumns, minicolumns, beta, w, p_pre=p, p_post=p, p_co=P,
               tau_z_post=tau_z_post, tau_z_pre=tau_z_pre,
               tau_a=tau_a, g_a=g_a, M=2, prng=prng)

    # Let's get the distances
    start = nn.o
    starting_point.append(start)

    # Calculate the closest pattern at the beginning
    distances_dic = calculate_distances_to_fix_points_dictionary(start, patterns)
    distances_history_start.append(distances_dic)
    closest_pattern_start.append(calculate_closest_pattern_dictionary(distances_dic))


    # Run the simulation and get the final equilibrum
    dic_history = nn.run_network_simulation(time, save=False)
    end = nn.o
    final_equilibrium.append(end)

    # Calculate the closest pattern at the end
    distances_dic = calculate_distances_to_fix_points_dictionary(end, patterns)
    distances_history_end.append(distances_dic)
    closest_pattern_end.append(calculate_closest_pattern_dictionary(distances_dic))


print(closest_pattern_end)
print(closest_pattern_start)