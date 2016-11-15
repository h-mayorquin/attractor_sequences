import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable

from data_transformer import transform_neural_to_normal
from analysis_functions import calculate_angle_from_history

sns.set(font_scale=1.0)


def plot_weight_matrix(nn, ampa=False):
    sns.set_style("whitegrid", {'axes.grid' : False})
    if ampa:
        w = nn.w_ampa
    else:
        w = nn.w

    aux_max = np.max(np.abs(w))

    cmap = 'coolwarm'
    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(111)
    im1 = ax1.imshow(w, cmap=cmap, interpolation='None', vmin=-aux_max, vmax=aux_max)

    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, ax=ax1, cax=cax1)


def plot_network_activity_angle(manager):

    T_total = manager.T_total
    history = manager.history
    # Get the angles
    angles = calculate_angle_from_history(manager)

    # Plot
    sns.set_style("whitegrid", {'axes.grid': False})

    cmap = 'plasma'
    extent1 = [0, manager.nn.minicolumns * manager.nn.hypercolumns, T_total, 0]
    extent2 = [0, manager.nn.minicolumns, T_total, 0]

    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(121)
    im1 = ax1.imshow(history['o'], aspect='auto', interpolation='None', cmap=cmap, vmax=1, vmin=0, extent=extent1)
    ax1.set_title('Unit activation')

    ax1.set_xlabel('Units')
    ax1.set_ylabel('Time')

    ax2 = fig.add_subplot(122)
    im2 = ax2.imshow(angles, aspect='auto', interpolation='None', cmap=cmap, vmax=1, vmin=0, extent=extent2)
    ax2.set_title('Winning pattern')

    ax2.set_xlabel('Patterns')


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.12, 0.05, 0.79])
    fig.colorbar(im1, cax=cbar_ax)


def plot_network_activity(manager):

    T_total = manager.T_total

    history = manager.history
    sns.set_style("whitegrid", {'axes.grid': False})

    cmap = 'plasma'
    extent = [0, manager.nn.minicolumns * manager.nn.hypercolumns, T_total, 0]

    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(221)
    im1 = ax1.imshow(history['o'], aspect='auto', interpolation='None', cmap=cmap, vmax=1, vmin=0, extent=extent)
    ax1.set_title('Unit activation')

    ax2 = fig.add_subplot(222)
    im2 = ax2.imshow(history['z_pre'], aspect='auto', interpolation='None', cmap=cmap, vmax=1, vmin=0, extent=extent)
    ax2.set_title('Traces of activity (z)')

    ax3 = fig.add_subplot(223)
    im3 = ax3.imshow(history['a'], aspect='auto', interpolation='None', cmap=cmap, vmax=1, vmin=0, extent=extent)
    ax3.set_title('Adaptation')

    ax4 = fig.add_subplot(224)
    im4 = ax4.imshow(history['p_pre'], aspect='auto', interpolation='None', cmap=cmap, vmax=1, vmin=0, extent=extent)
    ax4.set_title('Probability')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.12, 0.05, 0.79])
    fig.colorbar(im1, cax=cbar_ax)


def plot_adaptation_dynamics(manager, traces_to_plot):

    sns.set_style("darkgrid", {'axes.grid': True})
    history = manager.history
    minicolumns = manager.nn.minicolumns

    # Get the right time
    T_total = manager.T_total

    total_time = np.arange(0, T_total, manager.dt)

    # Extract the required data
    o_hypercolum = history['o'][..., :minicolumns]
    a_hypercolum = history['a'][..., :minicolumns]

    # Plot configuration
    cmap_string = 'Paired'
    cmap = matplotlib.cm.get_cmap(cmap_string)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=minicolumns)

    fig = plt.figure(figsize=(16, 12))

    ax11 = fig.add_subplot(221)
    ax12 = fig.add_subplot(222)
    ax21 = fig.add_subplot(223)
    ax22 = fig.add_subplot(224)

    fig.tight_layout()
    import IPython
    # IPython.embed()
    # Plot the wanted activities
    for index in traces_to_plot:
        ax11.plot(total_time, o_hypercolum[:, index], color=cmap(norm(index)), label=str(index))

    # Plot ALL the activities
    for index in range(minicolumns):
        ax12.plot(total_time, o_hypercolum[:, index], color=cmap(norm(index)), label=str(index))

    # Plot the wanted adaptations
    for index in traces_to_plot:
        ax21.plot(total_time, a_hypercolum[:, index], color=cmap(norm(index)), label=str(index))

    # Plot ALL the adaptations
    for index in range(minicolumns):
        ax22.plot(total_time, a_hypercolum[:, index], color=cmap(norm(index)), label=str(index))

    axes = fig.get_axes()
    for ax in axes:
        ax.set_xlim([0, T_total])
        ax.set_ylim([-0.1, 1.1])
        ax.legend()
        ax.axhline(0, color='black')

    ax11.set_title('Unit activity')
    ax21.set_title('Adaptations')


def plot_state_variables_vs_time(manager, traces_to_plot, ampa=False):

    sns.set_style("darkgrid", {'axes.grid': True})
    history = manager.history
    minicolumns = manager.nn.minicolumns

    T_total = manager.T_total

    total_time = np.arange(0, T_total, manager.dt)

    o_hypercolum = history['o'][..., :minicolumns]

    if ampa:
        z_pre_hypercolum = history['z_pre_ampa'][..., :minicolumns]
        z_post_hypercolum = history['z_post_ampa'][..., :minicolumns]

        p_pre_hypercolum = history['p_pre_ampa'][..., :minicolumns]
        p_post_hypercolum = history['p_post_ampa'][..., :minicolumns]

        # Take coactivations
        p_co = history['p_co_ampa']
        z_co = history['z_co_ampa']
        w = history['w_ampa']
    else:
        z_pre_hypercolum = history['z_pre'][..., :minicolumns]
        z_post_hypercolum = history['z_post'][..., :minicolumns]
        o_hypercolum = history['o'][..., :minicolumns]
        p_pre_hypercolum = history['p_pre'][..., :minicolumns]
        p_post_hypercolum = history['p_post'][..., :minicolumns]

        # Take coactivations
        p_co = history['p_co']
        z_co = history['z_co']
        w = history['w']

    # Build labels and pairs
    coactivations_to_plot = [(traces_to_plot[2], traces_to_plot[1]), (traces_to_plot[0], traces_to_plot[1])]
    labels_of_coactivations = [str(x) + '<--' + str(y) for (x, y) in coactivations_to_plot]

    p_co_list = []
    z_co_list = []
    w_list = []

    for (x, y) in coactivations_to_plot:
        p_co_list.append(p_co[:, x, y])
        z_co_list.append(z_co[:, x, y])
        w_list.append(w[:, x, y])


    cmap_string = 'nipy_spectral'
    cmap_string = 'hsv'
    cmap_string = 'Paired'
    cmap = matplotlib.cm.get_cmap(cmap_string)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=minicolumns)

    # Plot the traces
    fig = plt.figure(figsize=(20, 15))
    ax11 = fig.add_subplot(421)
    ax12 = fig.add_subplot(422)
    ax21 = fig.add_subplot(423)
    ax22 = fig.add_subplot(424)
    ax31 = fig.add_subplot(425)
    ax32 = fig.add_subplot(426)
    ax41 = fig.add_subplot(427)
    ax42 = fig.add_subplot(428)

    fig.tight_layout()

    for index in range(minicolumns):
        # Plot ALL the activities
        ax12.plot(total_time, o_hypercolum[:, index], label=str(index))

    for index in traces_to_plot:
        # Plot activities
        ax11.plot(total_time, o_hypercolum[:, index], color=cmap(norm(index)), label=str(index))

        # Plot the z post and pre traces in the same graph
        ax21.plot(total_time, z_pre_hypercolum[:, index], color=cmap(norm(index)), label='pre ' + str(index))
        ax21.plot(total_time, z_post_hypercolum[:, index], color=cmap(norm(index)), linestyle='--',
                  label='post ' + str(index))

        # Plot the pre and post probabilties in the same graph
        ax22.plot(total_time, p_pre_hypercolum[:, index], color=cmap(norm(index)), label='pre ' + str(index))
        ax22.plot(total_time, p_post_hypercolum[:, index], color=cmap(norm(index)), linestyle='--',
                  label='post ' + str(index))

    # Plot z_co and p_co in the same graph
    for z_co, label in zip(z_co_list, labels_of_coactivations):
        ax31.plot(total_time, z_co, label='z_co ' + label)

    # Plot the individual probabilities and the coactivations
    for p_co, (x, y), label in zip(p_co_list, coactivations_to_plot, labels_of_coactivations):
        ax32.plot(total_time, p_co, '-', label='p_co ' + label)
        ax32.plot(total_time, p_post_hypercolum[:, x] * p_pre_hypercolum[:, y],
                  label='p_post_' + label[0] + ' x p_pre_' + label[1])

    # Plot the coactivations probabilities
    for p_co, label in zip(p_co_list, labels_of_coactivations):
        ax41.plot(total_time, p_co, '-', label='p_co ' + label)

    # Plot the weights
    for w, label in zip(w_list, labels_of_coactivations):
        ax42.plot(total_time, w, label=r'$w_{' + label + '}$')

    axes = fig.get_axes()
    for ax in axes:
        ax.set_xlim([0, T_total])
        ax.legend()
        ax.axhline(0, color='black')

    ax11.set_ylim([-0.1, 1.1])
    ax12.set_ylim([-0.1, 1.1])
    ax21.set_ylim([-0.1, 1.1])
    ax31.set_ylim([-0.1, 1.1])

    ax21.set_title('z-traces')
    ax22.set_title('probabilities')
    ax31.set_title('z_co')
    ax32.set_title('p_co and p_i * p*j')
    ax41.set_title('p_co')
    ax42.set_title('w')


def plot_quantity_history(dic_history, quantity, minicolumns=2):

    sns.set_style("whitegrid", {'axes.grid': False})

    quantity_to_plot_1 = transform_neural_to_normal(dic_history[quantity], minicolumns=2)
    quantity_to_plot_2 = dic_history[quantity]

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
