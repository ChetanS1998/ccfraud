# This is a file reader that knows how to parse CSV files
# and turn them into useful data structures.
from numpy import genfromtxt


def read_to_numpy_array(file_path):
    '''
    Reads a file and transforms it into a numpy array.
    :param file_path: The path of the file
    :return: A numpy array
    '''
    return genfromtxt(file_path, delimiter=',')
