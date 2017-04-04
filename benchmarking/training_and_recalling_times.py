import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from standard_program import run_standard_program, calculate_succes_program, training_program
import timeit

sns.set(font_scale=2.0)

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

hypercolumns = 4
minicolumns = 10
epochs = 3
manager = run_standard_program(hypercolumns, minicolumns, epochs)

if True:
    T_recall_range = np.arange(3, 20, 1)
    time_recall = []

    for T_recall in T_recall_range:
        function = wrapper(training_program, manager=manager, T_recall=T_recall)
        time = timeit.timeit(function, number=1)
        time_recall.append(time)

    # Plot4
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.plot(T_recall_range, time_recall, '*-', markersize=4)

    ax.set_xlabel('T_recall')
    ax.set_ylabel('Seconds that the program took to run')
    ax.set_title('Normal recall profile')
    plt.show()

if True:
    T_recall_range = np.arange(3, 20, 1)
    time_success = []

    for T_recall in T_recall_range:
        function = wrapper(calculate_succes_program, manager=manager, T_recall=T_recall)
        time = timeit.timeit(function, number=1)
        time_success.append(time)

        # Plot
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.plot(T_recall_range, time_success, '*-', markersize=4)

    ax.set_xlabel('T_recall')
    ax.set_ylabel('Seconds that the program took to run')
    ax.set_title('Recall Success profiling')
    plt.show()