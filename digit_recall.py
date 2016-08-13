import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

from sklearn import datasets
from connectivity_functions import get_beta, get_w, softmax
from data_transformer import transform_normal_to_neural_single
from data_transformer import transform_neural_to_normal_single
from connectivity_functions import calculate_probability, calculate_coactivations


digits = datasets.load_digits()

images = digits['images']
data = digits['data']
target_names = digits['target_names']
target = digits['target']

# Let's binarize the image
data[data < 8] = 0
data[data >= 8] = 1

# Let's get two images and learn them as a pattern
number_of_patterns = 10
patterns = []
for i in range(number_of_patterns):
    patterns.append(data[i])

neural_patterns = [transform_normal_to_neural_single(pattern) for pattern in patterns]

p = calculate_probability(neural_patterns)
P = calculate_coactivations(neural_patterns)

w = get_w(P, p)
beta = get_beta(p)

# Here we have the evolution
T = 100
dt = 0.1
tau_m = 1.0
G = 1.0

o = np.random.rand(p.size)
# Save initial image
initial_image = np.copy(transform_neural_to_normal_single(o).reshape(8, 8))
m = np.zeros_like(o)


def update_system():
    global o, s, m
    # Update S
    s = beta + np.dot(w, o)
    # Evolve m, trailing s
    m += (dt / tau_m) * (s - m)
    # Softmax for m
    o = softmax(m, t=(1 / G))


for t in range(T):
    update_system()

final_image = transform_neural_to_normal_single(o).reshape(8, 8)

# Plot the two patterns and the final result
gs = gridspec.GridSpec(number_of_patterns, number_of_patterns, wspace=0.1, hspace=0.1)
cmap = cm.bone
interpolation = 'nearest'

fig = plt.figure(figsize=(16, 12))

for i in range(number_of_patterns):
    ax = fig.add_subplot(gs[0, i])
    im = ax.imshow(patterns[i].reshape(8, 8), cmap=cmap, interpolation=interpolation)
    ax.set_title('Pattern ' + str(i))
    ax.set_axis_off()


ax10 = fig.add_subplot(gs[1:, 0:(number_of_patterns/2)])
im10 = ax10.imshow(initial_image, cmap=cmap, interpolation=interpolation)
ax10.set_title('Initial Image')
ax10.set_axis_off()


ax11 = fig.add_subplot(gs[1:, (number_of_patterns/2):])
im11 = ax11.imshow(final_image, cmap=cmap, interpolation=interpolation)
ax11.set_title('Final image')
ax11.set_axis_off()


fig.show()
