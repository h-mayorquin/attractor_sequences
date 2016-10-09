from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from connectivity_functions import get_beta, get_w
from connectivity_functions import calculate_probability, calculate_coactivations
from data_transformer import build_ortogonal_patterns
from network import BCPNN

np.set_printoptions(suppress=True)

hypercolumns = 3
minicolumns = 3

patterns_dic = build_ortogonal_patterns(hypercolumns, minicolumns)
patterns = list(patterns_dic.values())

P = calculate_coactivations(patterns)
p = calculate_probability(patterns)

w = get_w(P, p)
beta = get_beta(p)

dt = 0.001
T_simulation = 10.0
T_training = 1.0
simulation_time = np.arange(0, T_simulation + dt, dt)
training_time = np.arange(0, T_simulation + dt, dt)

prng = np.random.RandomState(seed=0)

tolerance = 1e-5

g_a_set = np.arange(0, 110, 10)
g_beta_set = np.arange(0, 22, 2)
g_w_set = np.arange(0, 12, 2)

nn = BCPNN(hypercolumns, minicolumns, beta, w, g_a=1, g_beta=1.0, g_w=1.0, g_I=10.0, prng=prng)


if False:
    for pattern in reversed(patterns):
        I = pattern
        nn.randomize_pattern()
        print('---------')
        print('I')
        print(I)
        # Run the network for one minut clamped to the patternr
        nn.k = 1.0
        nn.run_network_simulation(training_time, I=I)
        # Run the network free
        print('---------')
        print('This should look like I')
        print(nn.o)
        nn.k = 0.0
        nn.run_network_simulation(training_time)
        print('---------')
        print('---------')
        print(nn.o)


if True:
    nn.randomize_pattern()
    nn.k = 1.0
    history = nn.run_network_simulation(training_time, save=True, I=patterns[0])
    nn.run_network_simulation(training_time, save=True, I=None)
    history = nn.run_network_simulation(training_time, save=True, I=patterns[1])
    history = nn.run_network_simulation(training_time, save=True, I=patterns[2])

    o = history['o']
    s = history['s']
    z_pre = history['z_pre']
    p_pre = history['p_pre']
    p_post = history['p_post']
    p_co = history['p_co']
    beta = history['beta']
    w = history['w']
    adaptation = history['a']

    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(221)
    im1 = ax1.imshow(o, aspect='auto', interpolation='None', cmap='viridis', vmax=1, vmin=0)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax1, ax=ax1)

    ax2 = fig.add_subplot(222)
    im2 = ax2.imshow(z_pre, aspect='auto', interpolation='None', cmap='viridis', vmax=1, vmin=0)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax2, ax=ax2)

    ax3 = fig.add_subplot(223)
    im3 = ax3.imshow(adaptation, aspect='auto', interpolation='None', cmap='viridis', vmax=1, vmin=0)
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax3, ax=ax3)

    ax4 = fig.add_subplot(224)
    im4 = ax4.imshow(p_pre, aspect='auto', interpolation='None', cmap='viridis', vmax=1, vmin=0)
    divider = make_axes_locatable(ax4)
    cax4 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im4, cax=cax4, ax=ax4)

    plt.show()

if False:
    nn.randomize_pattern()
    nn.k = 1.0

    history = nn.run_network_simulation(training_time, save=True, I=patterns[0])
    o = history['o']
    s = history['s']
    z_pre = history['z_pre']
    p_pre = history['p_pre']
    p_post = history['p_post']
    p_co = history['p_co']
    beta = history['beta']
    w = history['w']
    adaptation = history['a']

    history = nn.run_network_simulation(training_time, save=True, I=patterns[1])
    o = np.concatenate((o, history['o']))
    s = np.concatenate((s, history['s']))
    z_pre = np.concatenate((z_pre, history['z_pre']))
    p_pre = np.concatenate((p_pre, history['p_pre']))
    p_post = history['p_post']
    p_co = history['p_co']
    beta = history['beta']
    w = history['w']
    adaptation = np.concatenate((adaptation, history['a']))

    history = nn.run_network_simulation(training_time, save=True, I=patterns[2])
    o = np.concatenate((o, history['o']))
    s = np.concatenate((s, history['s']))
    z_pre = np.concatenate((z_pre, history['z_pre']))
    p_pre = np.concatenate((p_pre, history['p_pre']))
    p_post = history['p_post']
    p_co = history['p_co']
    beta = history['beta']
    w = history['w']
    adaptation = np.concatenate((adaptation, history['a']))

    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(221)
    im1 = ax1.imshow(o, aspect='auto', interpolation='None', cmap='viridis', vmax=1, vmin=0)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax1, ax=ax1)

    ax2 = fig.add_subplot(222)
    im2 = ax2.imshow(z_pre, aspect='auto', interpolation='None', cmap='viridis', vmax=1, vmin=0)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax2, ax=ax2)

    ax3 = fig.add_subplot(223)
    im3 = ax3.imshow(adaptation, aspect='auto', interpolation='None', cmap='viridis', vmax=1, vmin=0)
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax3, ax=ax3)

    ax4 = fig.add_subplot(224)
    im4 = ax4.imshow(p_pre, aspect='auto', interpolation='None', cmap='viridis', vmax=1, vmin=0)
    divider = make_axes_locatable(ax4)
    cax4 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im4, cax=cax4, ax=ax4)

    plt.show()