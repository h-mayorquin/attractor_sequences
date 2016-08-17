import numpy as np
from connectivity_functions import get_beta, get_w, softmax
from connectivity_functions import calculate_probability, calculate_coactivations
# np.set_printoptions(suppress=True)

pattern1 = np.array((1, 0, 1, 0))
pattern2 = np.array((0, 1, 1, 0))
patterns = [pattern1, pattern2]

P = calculate_coactivations(patterns)
p = calculate_probability(patterns)

w = get_w(P, p)
beta = get_beta(p)

# Here we have the evolution
T = 10
dt = 0.1

tau_m = 1.0
G = 1.0

o = np.random.rand(p.size)
m = np.zeros_like(o)

for t in range(T):
    # Update S
    s = beta + np.dot(w, o)
    # Evolve m, trailing s
    m += (dt / tau_m) * (s - m)
    # Softmax for m
    o = softmax(m, t=(1/ G))
