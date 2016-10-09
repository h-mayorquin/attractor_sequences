from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from connectivity_functions import get_beta, get_w, get_w_old, softmax
from connectivity_functions import calculate_probability, calculate_coactivations
from data_transformer import build_ortogonal_patterns
from network import BCPNN

np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)

hypercolumns = 3
minicolumns = 3

patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
patterns = list(patterns_dic.values())[0:4]


# The total probability
p = calculate_probability(patterns)
P = calculate_coactivations(patterns)

w = get_w(P, p)
x = np.outer(p, p)
w_aux = get_w_old(P, p)

# Plot it
cmap = 'coolwarm'
aux_max = np.max(np.abs(w))

fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(121)
# plt.imshow(w, cmap=cmap, interpolation='None', vmin=-aux_max, vmax=aux_max)
im1 = ax1.imshow(w, cmap=cmap, interpolation='None')
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, ax=ax1, cax=cax1)


ax2 = fig.add_subplot(122)
im2 = ax2.imshow(w_aux, cmap=cmap, interpolation='None')
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, ax=ax2, cax=cax2)

plt.show()