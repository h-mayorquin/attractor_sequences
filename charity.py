import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tau_m = 0.020   # ms
tau_z = 0.050  # ms

dt = 0.001
N = 2


x = np.array((1.0, 0))
z = np.array((0.0, 0.0))
w = np.array(([0, 1.0], [1.0, 0]))
input = 1

x_history = []
z_history = []

k = 1.0
def function(x):
    return 1 / (1 + np.exp(-k * x))

for i in range(10):
    within_activity = np.dot(w, z)
    # x += (dt / tau_m) * (function(within_activity - input) - x)
    increase = (dt / tau_z) * (x - z)
    print(increase)
    z += increase

    x_history.append(np.copy(x))
    z_history.append(np.copy(z))

print(x)
print(z)
# Plot the result
x_history = np.array(x_history)
z_history = np.array(z_history)
fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

for index, x_plot in enumerate(x_history.T):
    ax1.plot(x_plot, label=str(index))

for index, z_plot in enumerate(z_history.T):
    ax2.plot(z_plot,label=str(index))


ax1.set_ylim([0, 1.1])
ax2.set_ylim([0, 1.1])
ax1.legend()
ax2.legend()
plt.show()