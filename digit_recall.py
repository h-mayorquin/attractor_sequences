import numpy as np
from connectivity_functions import get_beta, get_w, softmax
from sklearn import datasets
from data_transformer import transform_normal_to_neural_single
from data_transformer import transform_neural_to_normal_single


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

p = 0.5 * (pattern1_neural + pattern2_neural)
P = 0.5 * (np.outer(pattern1_neural, pattern2_neural))

w = get_w(P, p)
beta = get_beta(p)

# Here we have the evolution
T = 1000
dt = 0.1
tau_m = 1.0
G = 1.0

o = np.random.rand()
m = np.zeros_like(o)

for t in range(T):
    # Update S
    s = beta + np.dot(w, o)
    # Evolve m, trailing s
    m += (dt / tau_m) * (s - m)
    # Softmax for m
    o = softmax(m, t=(1/ G))
