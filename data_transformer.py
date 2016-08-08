import numpy as np


def transform_normal_to_neural_single(normal, quantization_value=2):

    neural = np.zeros((normal.size, quantization_value))

    if True:
        for index, value in enumerate(normal):
            neural[index, value] = 1

    #transformed_input[:, input] = 1

    neural = neural.flatten()
    return neural


def transform_neural_to_normal_single(neural, quantization_value=2):

    normal = neural.reshape((neural.size/quantization_value), quantization_value)

    return normal[:, 1]


