import numpy as np
import itertools
import random
from network import Protocol

x = {1, 2, 3, 4}
y = {3, 4, 5, 6}

m = 50
k = 10
p = 4
o = 5

order = True
desired_sequences = 20
sequences = []
protocol = Protocol()
sequences, p_array, sequences_array, overlap_array = protocol.generate_sample_sequence(m, k, p, o, desired_sequences,
                                                                                       order=True, array_view=True, overlap_view=True)