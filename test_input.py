import numpy as np
import numpy.testing as npt
import unittest

from data_transformer import transform_normal_to_neural_single
from data_transformer import transform_neural_to_normal_single


class TestDataTransfomer(unittest.TestCase):
    def test_normal_to_neural_simplest(self):
        test_input_1 = np.array((1, 0, 1, 0))
        test_input_2 = np.array((0, 1, 0, 1))

        desired_1 = np.array((0, 1, 1, 0, 0, 1, 1, 0))
        desired_2 = np.array((1, 0, 0, 1, 1, 0, 0, 1))

        transform_1 = transform_normal_to_neural_single(test_input_1)
        transform_2 = transform_normal_to_neural_single(test_input_2)

        npt.assert_almost_equal(transform_1, desired_1)
        npt.assert_almost_equal(transform_2, desired_2)

    def test_neural_to_normal_simplest(self):
        test_input_1 = np.array((0, 1, 1, 0, 0, 1, 1, 0))
        test_input_2 = np.array((1, 0, 0, 1, 1, 0, 0, 1))

        desired_1 = np.array((1, 0, 1, 0))
        desired_2 = np.array((0, 1, 0, 1))

        transform_1 = transform_neural_to_normal_single(test_input_1)
        transform_2 = transform_neural_to_normal_single(test_input_2)

        npt.assert_almost_equal(transform_1, desired_1)
        npt.assert_almost_equal(transform_2, desired_2)

if __name__ == '__main__':
    unittest.main()