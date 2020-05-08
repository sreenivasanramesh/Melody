"""
Contains functions used by others
"""
import pickle


def read_binary_file(file_name):
    """reads binary file and returns the content"""
    with open(str(file_name), 'rb') as bin_file:
        obj = pickle.load(bin_file)
        return obj



