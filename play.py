import numpy as np
from connectivity_functions import get_beta, get_w, softmax
from connectivity_functions import calculate_probability, calculate_coactivations
from data_transformer import transform_normal_to_neural_single
from data_transformer import transform_neural_to_normal_single
from network import BCPNN
np.set_printoptions(suppress=True)

pattern1 = transform_normal_to_neural_single(np.array((1, 0, 0, 0, 0)))
pattern2 = transform_normal_to_neural_single(np.array((0, 0, 0, 0, 1)))
patterns = [pattern1, pattern2]

P = calculate_coactivations(patterns)
p = calculate_probability(patterns)

w = get_w(P, p)
beta = get_beta(p)

network = BCPNN(beta, w)
print(network.o)
network.update_discrete()
print(network.o)
network.update_discrete()
print(network.o)

x = transform_neural_to_normal_single(network.o)
