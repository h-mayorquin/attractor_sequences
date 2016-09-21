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


pattern1 = transform_normal_to_neural_single(np.array((1, 0, 1)))
pattern2 = transform_normal_to_neural_single(np.array((0, 1, 0)))
pattern3 = transform_normal_to_neural_single(np.array((0, 0, 1)))
patterns = [pattern1, pattern2, pattern3]

P = calculate_coactivations(patterns)
p = calculate_probability(patterns)

w = get_w(P, p)
beta = get_beta(p)

# Parameters
tau_z_post = 0.240
tau_z_pre = 0.240
g_a = 80
tau_a = 2.7

nn = BCPNN(beta, w, p_pre=p, p_post=p, p_co=P, tau_z_post=tau_z_post, tau_z_pre=tau_z_pre,
           tau_a=tau_a, g_a=g_a, M=2)

dt = 0.01
T = 10
time = np.arange(0, T + dt, dt)

history_o = np.zeros((time.size, beta.size))
history_s = np.zeros_like(history_o)
history_z_pre = np.zeros_like(history_o)
history_z_post = np.zeros_like(history_o)
history_a = np.zeros_like(history_o)

for index_t, t in enumerate(time):
    nn.update_continuous(dt)
    history_o[index_t, :] = nn.o
    history_s[index_t, :] = nn.s
    history_z_pre[index_t, :] = nn.z_pre
    history_z_post[index_t, :] = nn.z_post
    history_a[index_t, :] = nn.a

x = transform_neural_to_normal_single(nn.o)

# Plotting goes here
quantity_to_plot = transform_neural_to_normal(history_o)

sns.set_style("whitegrid", {'axes.grid' : False})

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111)
im = ax.imshow(quantity_to_plot, aspect='auto', interpolation='nearest')

divider1 = make_axes_locatable(ax)
cax1 = divider1.append_axes("right", size='5%', pad=0.05)
fig.colorbar(im, cax=cax1)

plt.show()


