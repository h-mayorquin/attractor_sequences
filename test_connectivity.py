import numpy as np
import numpy.testing as npt
import unittest

from data_transformer import transform_normal_to_neural_single
from data_transformer import transform_neural_to_normal_single
from data_transformer import transform_neural_to_normal
from connectivity_functions import calculate_probability, calculate_coactivations


class TestDataTransformer(unittest.TestCase):
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

    def test_neural_to_normal(self):
        test_input = np.array(((0, 1, 1, 0, 0, 1), (1, 0, 0, 1, 0, 1), (0, 1, 1, 0, 0, 1)))
        desired = np.array(((1, 0, 1), (0, 1, 1), (1, 0, 1)))

        transform = transform_neural_to_normal(test_input)

        npt.assert_almost_equal(transform, desired)


class TestProbabilities(unittest.TestCase):
    def test_unit_probabilities(self):
        test_pattern1 = np.array((1, 0, 1, 0))
        test_pattern2 = np.array((0, 1, 0, 1))

        patterns = [test_pattern1, test_pattern2]

        desired_probability = np.array((0.5, 0.5, 0.5, 0.5))

        calculated_probability = calculate_probability(patterns)

        npt.assert_almost_equal(calculated_probability, desired_probability)

    def test_coactivations(self):
        test_pattern1 = np.array((1, 0, 1, 0))
        test_pattern2 = np.array((0, 1, 0, 1))

        patterns = [test_pattern1, test_pattern2]

        desired_coactivations = np.array(((0.5, 0, 0.5, 0), (0, 0.5, 0, 0.5),
                                         (0.5, 0, 0.5, 0), (0, 0.5, 0, 0.5)))

        calculated_coactivations = calculate_coactivations(patterns)

        npt.assert_almost_equal(desired_coactivations, calculated_coactivations)


if __name__ == '__main__':
    unittest.main()