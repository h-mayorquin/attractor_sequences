import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
pattern1 = data[0]
pattern2 = data[1]

pattern1_neural = transform_normal_to_neural_single(pattern1)
pattern2_neural = transform_normal_to_neural_single(pattern2)

patterns = [pattern1_neural, pattern2_neural]

p_aux = 0.5 * (pattern1_neural + pattern2_neural)
P_aux = 0.5 * (np.outer(pattern1_neural, pattern1_neural) + np.outer(pattern2_neural, pattern2_neural))

p = calculate_probability(patterns)
P = calculate_coactivations(patterns)

w = get_w(P, p)
beta = get_beta(p)

# Here we have the evolution
T = 1000
dt = 0.1
tau_m = 1.0
G = 1.0

o = np.random.rand(p.size)
# Save initial image
initial_image = np.copy(transform_neural_to_normal_single(o).reshape(8, 8))
m = np.zeros_like(o)

for t in range(T):
    # Update S
    s = beta + np.dot(w, o)
    # Evolve m, trailing s
    m += (dt / tau_m) * (s - m)
    # Softmax for m
    o = softmax(m, t=(1/ G))


# Plot the two patterns and the final result
gs = gridspec.GridSpec(2, 2)
fig = plt.figure(figsize=(16, 12))

ax00 = fig.add_subplot(gs[0, 0])
ax00.imshow(pattern1.reshape(8, 8))

ax01 = fig.add_subplot(gs[0, 1])
ax01.imshow(pattern2.reshape(8, 8))

ax10 = fig.add_subplot(gs[1, 0])
ax10.imshow(initial_image)

ax11 = fig.add_subplot(gs[1, 1])
ax11.imshow(transform_neural_to_normal_single(o).reshape(8, 8))

fig.show()
