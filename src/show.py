from preprocessing.data import get_data_patient, get_data
from visualization.data_visualizer import plot_data, DataVisualizer
from visualization.log import log_results
from optparse import OptionParser
from matplotlib import pyplot as plt
import numpy as np
import os
import inspect

def plot_raw(*, data_dir='data'):
    print('Has to be implemented')


def plot_samples(*, data_dir='data', patch_num=10, patient_num='0023',
                 tumor_rate=0.5, data_set='train', **rest):
    """
    Plots samples from the dataset. Can be used to get the data for a specific
    patient or to plot samples from a whole dataset.

    :param patient_num: Can contain a specific patient number
    :param data_set: specify the dataset to be used here (either 'train' or 'test')
    """
    if patient_num is not None:
        train_data, labels = get_data_patient(data_dir,
                                              patient_num='LIDC-IDRI-' + str(patient_num), dir='train')
    else:
        train_data, labels = get_data(data_dir, data_set,
                                      patch_number=patch_num, tumor_rate=tumor_rate)

    #plot_data(train_data, labels)
    DataVisualizer(train_data, labels)
    plt.show()

def plot_network(*, data_dir='data'):
    print('Has to be implemented')

############################################################################
### Call and commandline helper functions

methods = { 'r':plot_raw,
            'p':plot_samples,
            'n':plot_network}

def get_commandline_args():
    """
    Creates the command line arguments and returns their values.
    """
    usage = "usage: python3 %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("-t", "--target", dest="target", choices=list(methods.keys()))
    parser.add_option("-d", "--data", dest="data_dir", help="path to the data directory")
    parser.add_option("-p", "--patientNum", dest="patient_num",
                      default=None, help="The patient number to be plotted from")
    parser.add_option("-n", "--number", dest="patch_num", type="int",
                      default=6, help="How many patches should be plotted (has to be even)")
    parser.add_option("-s", "--data_set", dest="data_set",
                      default="train", help="Plot from either train/test or validate")
    return parser.parse_args()


def get_vals(dictionary, *keys):
    """
    Creates an dictionary composed of the picked dictionary keys.
    >>> d = {'a': 1, 'b': 2, 'c': 3}
    >>> pick(d, 'b', 'a')
    {'b': 2, 'a': 1}
    """
    dict_keys = dictionary.keys()
    return {key: dictionary[key] for key in keys if key in dict_keys}

if __name__ == "__main__":
    (option, args) = get_commandline_args()

    target = option.target

    arg_names = list(inspect.signature(methods[target]).parameters)
    matching_params = get_vals(vars(option), *arg_names)
    methods[target](**matching_params)
