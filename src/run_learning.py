from preprocessing.data import get_train_data
from visualization.network_visualiser import plot_loss, plot_sample
from learning.network import train_network

import numpy as np


BATCH_SIZE = 20

PATCH_SIZE = [20, 20, 3]

if __name__ == "__main__":
    # get the data
    train_data_raw, train_labels_raw = get_train_data(patch_number=20, patch_size=PATCH_SIZE)

    train_data_raw = np.asarray(train_data_raw)
    train_labels_raw = np.asarray(train_labels_raw)

    plot_sample(train_data_raw, train_labels_raw)

    epochs_val, losses = train_network(train_data_raw, train_labels_raw)

    #saveResults()
    plot_loss(epochs_val, losses)