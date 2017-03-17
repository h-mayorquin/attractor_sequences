from __future__ import print_function
from pprint import pprint
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# parameters
m = 10
k = 4
o = 2
p = 2

verbose = True

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

while position + k < m:
    if verbose:
        print('********')
        print(position)
        print(position + k)
        print('**********')
        # Lay another sequence
    array_line = np.zeros(m, dtype='int')
    # Create overlap dic with as many keys as sequences we have stored so far
    overlap_dic = {index:0 for index in range(0, capacity)}

    # Move the position to the first position where p_indicator > 0
    for p_index, p_value in enumerate(p_indicator):
        if p_value > 0:
            position = p_index
            break

    if position + k >= m:
        print('break the while')
        break

    position_i = position
    for counter in range(k):

        # Check that there is still p left to add
        p_flag = True
        while p_flag:
            if p_indicator[position_i] == 0:
                if verbose:
                    print('p flag')
                position_i += 1
            else:
                p_flag = False

        # Check overlap
        overlap_flag = True
        while overlap_flag:
            for row in range(capacity):
                # If overlap is bigger than o and that sequence is already there skip
                if verbose:
                    print('--------------')
                    print('row', row)
                    print('position', position_i)
                    print('overlap_dic', overlap_dic)
                    print('array[row, position]', array[row, position_i])
                if (overlap_dic[row] >= o) and (array[row, position_i] == 1):
                    if verbose:
                        print('overlap flag')
                        print('--------------')
                    position_i += 1
                    overlap_flag = True
                    break
                else:
                    overlap_flag = False

        # Add the element
        array_line[position_i] += 1
        p_indicator[position_i] -= 1
        position_i += 1

        # Add overlap
        for row in range(capacity):
            # If there is an element
            if array[row, position_i] == 1:
                overlap_dic[row] += 1

    capacity += 1
    array = np.vstack((array, array_line))
    pprint(array)