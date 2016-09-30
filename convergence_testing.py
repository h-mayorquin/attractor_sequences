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

hypercolumns = 10
minicolumns = 10

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

N = 100

prng = np.random.RandomState(seed=0)


from convergence_functions import append_distances_history

# Run and extract data
for i in range(N):
    nn = BCPNN(hypercolumns, minicolumns, beta, w, p_pre=p, p_post=p, p_co=P,
               tau_z_post=tau_z_post, tau_z_pre=tau_z_pre,
               tau_a=tau_a, g_a=g_a, M=2, prng=prng)

    # Let's get the distances
    start = nn.o
    starting_point.append(start)

    # Calculate the closest pattern at the beginning
    append_distances_history(start, patterns, closest_pattern_start,
                             distances_history_start)

    # Run the simulation and get the final equilibrum
    nn.run_network_simulation(time, save=False)
    end = nn.o
    final_equilibrium.append(end)

    # Calculate the closest pattern at the end
    append_distances_history(end, patterns, closest_pattern_end,
                             distances_history_end)

# Let;s calculate how many patterns ended up in the fix points
tolerance = 1e-10
fraction_of_convergence = 0
for distances_end in distances_history_end:
    minimal_distance = min(distances_end.values())
    if minimal_distance < tolerance:
        fraction_of_convergence += 1

fraction_of_convergence = fraction_of_convergence * 1.0 / N
print('Fraction of convergence patterns', fraction_of_convergence)

# Let's calculate how many of the patterns ended up in the one that they started closer too
fraction_of_well_behaviour = [end - start for start, end in zip(closest_pattern_start, closest_pattern_end)].count(0)
fraction_of_well_behaviour  = fraction_of_well_behaviour * 1.0 / N

print('Fraction of well behaved patterns', fraction_of_well_behaviour)