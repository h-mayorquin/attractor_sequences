import sys
sys.path.append('../')

import numpy as np
from network import BCPNNFast, NetworkManager

# Patterns parameters
hypercolumns = 4
minicolumns = 20

# Manager properties
dt = 0.001
running_time = 20
values_to_save = []

# Build the network
nn = BCPNNFast(hypercolumns, minicolumns)
nn.k_inner = False
nn.k = 0.0

# Build the manager
manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)
time = np.arange(0, running_time, dt)
manager.run_network(time, I=0)