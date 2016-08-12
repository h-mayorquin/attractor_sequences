import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import seaborn as sns


e_minimum = 1e-10
n_points = 1000

px = np.linspace(e_minimum, 1, n_points)
py = np.linspace(e_minimum, 1, n_points)
pxy = np.linspace(e_minimum, 1, n_points)

py_constant = 1.0

# First case
aux = np.log2(np.outer(pxy, (1.0 /px)))
cmap = cm.coolwarm
number_of_levels = 50
levels = np.linspace(-10, 10, number_of_levels)

fig = plt.figure(figsize=(16 , 12))
ax = fig.add_subplot(111)
contour = ax.contourf(px, pxy, aux, cmap=cmap, levels=levels)
fig.colorbar(contour)

plt.show()