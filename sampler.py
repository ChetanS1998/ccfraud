# This mode splits the data into training and validation sets.

import numpy as np
def split_into_training_and_validation(data, ratio):
    '''
    Randomly splits a dataset into a training and validation sets
    :param data: A numpy ndarray
    :param ratio: A float, 0 < x < 1. A value of 0.85 means that 85% of the data
    will be used to train and 15% to validate
    :return: A tuple of two numpy arrays, the first is training and the second validation.
    '''

    # Ensure that the ratio is valid
    if ratio <= 0 or ratio > 1.:
        raise ValueError('The ratio must be number greater than 0 and less than 1')

    # Ensure we have data!
    if data is None:
        raise ValueError('You must provide a valid numpy array')

    # Ensure correct type, only ndarrays supported
    if type(data)!=np.ndarray:
        raise ValueError('You must provide a valid numpy array')

    percentage = int(len(data) * ratio)
    np.random.shuffle(data)
    return (data[:percentage, :], data[percentage:, :])