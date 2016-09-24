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


def transform_neural_to_normal(neural_matrix, quantization_value=2):
    """
    Transforms a matrix from the neural representation to the neural one

    :param neural_matrix: the neural representation
    :param quantization_value:  the number of values that each element is quantized

    :return: the normal matrix representation
    """

    number_of_elements, number_of_units = neural_matrix.shape

    normal_matrix = np.zeros((number_of_elements, number_of_units / quantization_value))

    for index, neural in enumerate(neural_matrix):
        normal_matrix[index, :] = transform_neural_to_normal_single(neural_matrix[index, :], quantization_value)

    return normal_matrix







