from preprocessing.data import get_train_data, get_validation_data
from tools.data_visualizer import DataVisualizer
from learning.network import train_network
from optparse import OptionParser
import logging.config
import numpy as np
import os


def get_commandline_args():
    """
    Creates the command line arguments and returns their values.
    """
    usage = "usage: python3 %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("-d", "--data", dest="data_dir", help="path to the data directory")
    parser.add_option("-e", "--epochs", dest="epochs", default=100,
                      help="number of training epochs", type="int")
    parser.add_option("-b", "--batchsize", dest="batchsize", default=20,
                      help="number of samples per batch", type="int")
    parser.add_option("-p", "--patch_number", dest="patch_number", type="int",
                      default=4000, help="Number of patches to be loaded from train")
    parser.add_option("-l", "--log_path", dest="log",
                      default=".", help="The directory to which the logged results shall be printed")
    parser.add_option("-s", "--save_level", dest="save_level",
                      default=100, help="At how many epochs the performance of the network is saved",
                      type="int")
    parser.add_option("-n", "--net_save_path", dest="net_save_path",
                      default='acts_net.tf', help="Path to the network storage location")
    parser.add_option("-t", "--test_name", dest="test_name",
                      default='exp', help="Description for this run")

    return parser.parse_args()


if __name__ == "__main__":
    # initialize logging
    logging.config.fileConfig(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + 'logging.ini')

    logger = logging.getLogger()

    (option, args) = get_commandline_args()

    if option.data_dir is None:
        raise ValueError('data_dir must be set to the correct folder path, use -d to specify the data location!')

    trains = option.patch_number
    vals = 0.1*option.patch_number
    logger.info('Reading in the Lung CT data, %d training and %d validation', trains, vals)
    train_data_raw, train_labels_raw = get_train_data(option.data_dir, patch_number=trains)
    validation_data_raw, validation_labels_raw = get_validation_data(option.data_dir, patch_number=vals)
    logger.info('Finished reading data')

    train_data_raw = np.asarray(train_data_raw)
    train_labels_raw = np.asarray(train_labels_raw)

    validation_data_raw = np.asarray(validation_data_raw)
    validation_labels_raw = np.asarray(validation_labels_raw)

    logger.info('Start training of the network')
    epochs_val, losses = train_network(train_data_raw, train_labels_raw,
                                       validation_data_raw, validation_labels_raw,
                                       batch_size=option.batchsize,
                                       epochs=option.epochs,
                                       save_level=option.save_level,
                                       net_save_path=option.net_save_path)

    logger.info('Finished training! Check out the log directory for results')
