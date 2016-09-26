import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable

from data_transformer import transform_neural_to_normal

sns.set(font_scale=2.0)


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
