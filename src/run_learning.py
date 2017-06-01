from preprocessing.data import get_train_data
from visualization.network_visualiser import plot_loss, plot_sample
from visualization.log import log_results
from learning.network import train_network
from optparse import OptionParser
import logging
import logging.config
import numpy as np
import os

PATCH_SIZE = [20, 20, 3]

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
    parser.add_option("-p", "--plotSample", dest="plot_samp",
                      default=False, help="True if a sample of the data should be plotted")
    parser.add_option("-l", "--logPath", dest="log",
                      default=".", help="The directory to which the log shall be printed")
    parser.add_option("-s", "--save_level", dest="save_level",
                      default=100, help="At how many epochs the performance of the network is saved",
                      type="int")

    return parser.parse_args()


if __name__ == "__main__":
    # initialize logging
    print(os.path.dirname(os.path.realpath(__file__)))

    logging.config.fileConfig(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + 'logging.ini')
    logger = logging.getLogger()

    (option, args) = get_commandline_args()

    if option.data_dir is None:
        raise ValueError('data_dir must be set to the correct folder path, use -d to specify the data location!')

    logger.info('Reading in the Lung CT data')
    train_data_raw, train_labels_raw = get_train_data(option.data_dir, patch_number=20, patch_size=PATCH_SIZE)
    logger.info('Finished reading data')

    train_data_raw = np.asarray(train_data_raw)
    train_labels_raw = np.asarray(train_labels_raw)

    if option.plot_samp:
      plot_sample(train_data_raw, train_labels_raw)

    logger.info('Start training of the network')
    epochs_val, losses = train_network(train_data_raw, train_labels_raw, batch_size=option.batchsize, epochs=option.epochs, save_level=option.save_level)
    logger.info('Finished training! Saving results...')

    log_results(epochs_val, losses, log_path = option.log)

    if option.plot_samp:
      plot_loss(epochs_val, losses)
