from preprocessing.data import get_train_data_patient, get_data
from visualization.data_visualizer import plot_data, DataBrowser
from visualization.log import log_results
from optparse import OptionParser
import numpy as np
import os


def plot_samples(*, data_dir='data', patch_num=10, patient_num='0023',
                 tumor_rate=0.5, data_set='train'):
    """
    Plots samples from the dataset. Can be used to get the data for a specific
    patient or to plot samples from a whole dataset.

    :param patient_num: Can contain a specific patient number
    :param data_set: specify the dataset to be used here (either 'train' or 'test')
    """
    if patient_num is not None:
        train_data, labels = get_train_data_patient(data_dir,
                                                patient_num='LIDC-IDRI-' + str(patient_num),
                                                patch_number=patch_num, tumor_rate=tumor_rate)
    else:
        train_data, labels = get_data(data_dir, data_set, patient_num=None,
                                      patch_number=patch_num, tumor_rate=tumor_rate)


    #plot_data(train_data, labels)
    DataBrowser(train_data, labels)


def get_commandline_args():
    """
    Creates the command line arguments and returns their values.
    """
    usage = "usage: python3 %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("-d", "--data", dest="data_dir", help="path to the data directory")
    parser.add_option("-p", "--patientNum", dest="patient_num",
                      default=None, help="The patient number to be plotted from")
    parser.add_option("-n", "--number", dest="patch_num", type="int",
                      default=6, help="How many patches should be plotted (has to be even)")
    parser.add_option("-s", "--data_set", dest="data_set",
                      default="train", help="Plot from either train/test or validate")
    return parser.parse_args()


if __name__ == "__main__":
    (option, args) = get_commandline_args()
    plot_samples(data_dir=option.data_dir, patch_num=option.patch_num,
                 patient_num=option.patient_num, data_set=option.data_set)
