import unittest
from sampler import split_into_training_and_validation
import numpy as np
import logging

class TestSplit(unittest.TestCase):

    def generate_sample(self):
        return np.array([[1000., 1000.1, 1000.2, 1000.3],
                         [2000., 2000.1, 2000.2, 2000.4],
                         [3000., 3000.1, 3000.2, 3000.4],
                         [4000., 4000.1, 4000.2, 4000.4],
                         [5000., 5000.1, 5000.2, 5000.4],
                         [6000., 6000.1, 6000.2, 6000.4],
                         [7000., 7000.1, 7000.2, 7000.4],
                         [8000., 8000.1, 8000.2, 8000.4],
                         [9000., 9000.1, 9000.2, 9000.4],
                         [10000., 10000.1, 10000.2, 10000.4]])

    def test_split_into_training_and_validation_bad_ratio(self):

        sample = self.generate_sample()

        # validate negative ratio
        with self.assertRaises(ValueError):
            (training, validation) = split_into_training_and_validation(sample, -1.)

        # validate zero ratio
        with self.assertRaises(ValueError):
            (training, validation) = split_into_training_and_validation(sample, 0.)

        # validate bigger than 1 ratio
        with self.assertRaises(ValueError):
            (training, validation) = split_into_training_and_validation(sample, 1.0001)

    def test_split_into_training_and_validation_bad_data(self):

        # validate empty data
        with self.assertRaises(ValueError):
            split_into_training_and_validation(None, 0.5)

        # validate wrong type of data
        with self.assertRaises(ValueError):
            split_into_training_and_validation([], 0.5)

    def test_split_into_training_and_validation(self):

        # validate proper split
        sample = self.generate_sample()
        (a, b) = split_into_training_and_validation(sample, 0.8)
        self.assertEqual(len(a), 8)
        self.assertEqual(len(b), 2)


if __name__=='__main__':
    unittest.main()