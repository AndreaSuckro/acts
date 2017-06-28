from preprocessing.data import get_train_data_patient
from visualization.data_visualizer import plot_data
from visualization.log import log_results
from optparse import OptionParser
import numpy as np
import os


def plot_sample(*, data_dir='data', patch_num=10, patient_num='0023', tumor_rate=0.5):
    train_data, labels = get_train_data_patient(data_dir,
                                                patient_num='LIDC-IDRI-' + str(patient_num),
                                                patch_number=patch_num, tumor_rate=tumor_rate)
    plot_data(train_data, labels)


def get_commandline_args():
    """
    Creates the command line arguments and returns their values.
    """
    usage = "usage: python3 %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("-d", "--data", dest="data_dir", help="path to the data directory")
    parser.add_option("-p", "--patientNum", dest="patient_num",
                      default='0023', help="The patient number to be plotted from")
    parser.add_option("-n", "--number", dest="patch_num", type="int",
                      default=6, help="How many patches should be plotted (has to be even)")

    return parser.parse_args()


if __name__ == "__main__":
    (option, args) = get_commandline_args()
    plot_sample(data_dir=option.data_dir, patch_num=option.patch_num, patient_num=option.patient_num)
