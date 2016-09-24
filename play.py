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


pattern1 = transform_normal_to_neural_single(np.array((1, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
pattern2 = transform_normal_to_neural_single(np.array((0, 1, 0, 0, 0, 0, 0, 0, 0, 0)))
pattern3 = transform_normal_to_neural_single(np.array((0, 0, 1, 0, 0, 0, 0, 0, 0, 0)))
pattern4 = transform_normal_to_neural_single(np.array((0, 0, 0, 1, 0, 0, 0, 0, 0, 0)))
pattern5 = transform_normal_to_neural_single(np.array((0, 0, 0, 0, 1, 0, 0, 0, 0, 0)))
pattern6 = transform_normal_to_neural_single(np.array((0, 0, 0, 0, 0, 1, 0, 0, 0, 0)))
pattern7 = transform_normal_to_neural_single(np.array((0, 0, 0, 0, 0, 0, 1, 0, 0, 0)))
pattern8 = transform_normal_to_neural_single(np.array((0, 0, 0, 0, 0, 0, 0, 1, 0, 0)))
pattern9 = transform_normal_to_neural_single(np.array((0, 0, 0, 0, 0, 0, 0, 0, 1, 0)))
pattern10 = transform_normal_to_neural_single(np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 1)))

patterns = [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7, pattern8, pattern9, pattern10]
patterns = [pattern1, pattern2, pattern3]

pattern1 = transform_normal_to_neural_single(np.array((1, 0, 0)))
pattern2 = transform_normal_to_neural_single(np.array((0, 1, 0)))
pattern3 = transform_normal_to_neural_single(np.array((0, 0, 1)))

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

N = 10

# Run and extract data
for i in range(N):
    nn = BCPNN(beta, w, p_pre=p, p_post=p, p_co=P, tau_z_post=tau_z_post, tau_z_pre=tau_z_pre,
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
    final_equilibrium.append(transform_neural_to_normal_single(nn.o))



# Plotting goes here
#from plotting_functions import plot_quantity_history
#plot_quantity_history(dic_history, 'o')\