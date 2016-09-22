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
T = 5
time = np.arange(0, T + dt, dt)

dic_history = nn.run_network_simulation(time, save=True)

# Retrieve the histories

# Plotting goes here
quantity_to_plot_1 = transform_neural_to_normal(dic_history['o'])
quantity_to_plot_2 = dic_history['o']

sns.set_style("whitegrid", {'axes.grid' : False})

gs = gridspec.GridSpec(1, 2)

fig = plt.figure(figsize=(16, 12))
ax1 =  fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(quantity_to_plot_1, aspect='auto', interpolation='nearest')

divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size='5%', pad=0.05)
fig.colorbar(im1, cax=cax1)

ax2 =  fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(quantity_to_plot_2, aspect='auto', interpolation='nearest')

divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size='5%', pad=0.05)
fig.colorbar(im2, cax=cax2)


plt.show()


