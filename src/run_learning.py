from preprocessing.data import get_train_data
from visualization.network_visualiser import plot_loss, plot_sample
from learning.network import train_network
from optparse import OptionParser

import numpy as np


PATCH_SIZE = [20, 20, 3]

if __name__ == "__main__":
    # parse commandline options
    usage = "usage: python3 %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("-d", "--data", dest="data_dir", help="path to the data directory")
    parser.add_option("-e", "--epochs", dest="epochs", default=100, help="number of training epochs")
    parser.add_option("-b", "--batchsize", dest="batchsize", default=20, help="number of samples per batch")
    parser.add_option("-p", "--plotSample", dest="plot_samp",
                      default=False, help="True if a sample of the data should be plotted")

    (option, args) = parser.parse_args()

    # get the data
    train_data_raw, train_labels_raw = get_train_data(option.data_dir, patch_number=20, patch_size=PATCH_SIZE)

    train_data_raw = np.asarray(train_data_raw)
    train_labels_raw = np.asarray(train_labels_raw)

    if option.plot_samp:
       plot_sample(train_data_raw, train_labels_raw)

    epochs_val, losses = train_network(train_data_raw, train_labels_raw, batch_size=option.batchsize, epochs=option.epochs)

    #saveResults()
    #plot_loss(epochs_val, losses)
