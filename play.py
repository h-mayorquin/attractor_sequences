from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from connectivity_functions import softmax

np.set_printoptions(suppress=True)

x = 0
points = np.logspace(0, 4, num=1000)
distance = np.zeros_like(points)

for index, y in enumerate(points):
    print(y)
    array = np.array((x, y))
    aux = softmax(array, minicolumns=2)
    distance[index] = np.linalg.norm(aux[0] - aux[1])

plt.plot(points, distance)