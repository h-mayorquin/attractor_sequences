from __future__ import print_function
from pprint import pprint
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# parameters
m = 17
k = 5
o = 3
p = 3

p_indicator = np.ones(m) * p
position = 0
capacity = 0
array = np.array([]).reshape(0, m)
# while(position < m):

# Lay the first sequence
array_line = np.zeros(m, dtype='int')
for i in range(k):
    array_line[position + i] = 1
    p_indicator[position + i] -= 1

# Increase capacity
capacity += 1
# Stack the array
array = np.vstack((array, array_line))


# Lay the second sequence
array_line = np.zeros(m, dtype='int')
# Create overlap dic with as many keys as sequences we have stored so far
overlap_dic = {index:0 for index in range(0, capacity)}

# Move the position to the first position where p_indicator > 0
for p_index, p_value in enumerate(p_indicator):
    if p_value > 0:
        position = p_index
        break

for position_i in range(k):

    # Check that there is still p left to add
    if p_indicator[position] == 0:
        print('p flag')
        position += 1

    # Check overlap
    for row in range(capacity):
        # If overlap is bigger than o and that sequence is already there skip
        if (overlap_dic[row] > o) and (array[row, position] == 1):
            print('overlap flag')
            position += 1
            break

    # Add the element
    array_line[position + position_i] += 1

    # Add overlap
    for row in range(capacity):
        # If there is an element
        if array[row, position] == 1:
            overlap_dic[row] += 1

capacity += 1
array = np.vstack((array, array_line))
pprint(array)