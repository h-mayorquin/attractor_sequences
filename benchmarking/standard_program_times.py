import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from standard_program import run_standard_program
import timeit

sns.set(font_scale=2.0)

def wrapper(func, hypercolumns, minicolumns, epochs):
    def wrapped():
        return func(hypercolumns, minicolumns, epochs)
    return wrapped

minicolumns_benchmark = True
hypercolumns_benchmark = True
epochs_benchmark = True

if minicolumns_benchmark:
    # Minicolumns
    hypercolumns = 4
    minicolumns_range = np.arange(10, 100, 5)
    epochs = 1

    times_minicolumns = []
    for minicolumns in minicolumns_range:
        function = wrapper(run_standard_program, hypercolumns, minicolumns, epochs)
        time = timeit.timeit(function, number=1)
        times_minicolumns.append(time)

    # Plot

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.plot(minicolumns_range, times_minicolumns, '*-', markersize=4)
    ax.set_xlabel('Minicolumns')
    ax.set_ylabel('Seconds that the program runed')

# Hypercolumns
if hypercolumns_benchmark:
    hypercolumns_range = np.arange(4, 20, 2)
    minicolumns = 20
    epochs = 1

    times_hypercolumns = []
    for hypercolumns in hypercolumns_range:
        function = wrapper(run_standard_program, hypercolumns, minicolumns, epochs)
        time = timeit.timeit(function, number=1)
        times_hypercolumns.append(time)

    # Plot
    sns.set(font_scale=2.0)
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.plot(hypercolumns_range, times_hypercolumns, '*-', markersize=4)
    ax.set_xlabel('Hypercolumns')
    ax.set_ylabel('Seconds that the program runed')

# Epochs
if epochs_benchmark:
    hypercolumns = 4
    minicolumns = 20
    epochs_range = np.arange(1, 10, 1)

    times_epochs = []
    for epochs in epochs_range:
        function = wrapper(run_standard_program, hypercolumns, minicolumns, epochs)
        time = timeit.timeit(function, number=1)
        times_epochs.append(time)

    # Plot
    sns.set(font_scale=2.0)
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.plot(epochs_range, times_epochs, '*-', markersize=4)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Seconds that the program runed')

if minicolumns_benchmark and hypercolumns_benchmark and epochs_benchmark:
    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.plot(minicolumns_range, times_minicolumns, '*-', markersize=4)
    ax2.plot(hypercolumns_range, times_hypercolumns, '*-', markersize=4)
    ax3.plot(epochs_range, times_epochs, '*-', markersize=4)

    ax1.set_title('Minicolumn scaling')
    ax2.set_title('Hypercolumn scaling')
    ax3.set_title('Epoch scaling')

    ax1.set_ylabel('Time ')

if minicolumns_benchmark or hypercolumns_benchmark or epochs_benchmark:
    plt.show()