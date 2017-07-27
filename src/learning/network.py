import tensorflow as tf
import numpy as np
import logging
from datetime import datetime
from visualization.log import log_args
import os

from learning.network_structure import network_model


@log_args
def train_network(train_data, train_labels, test_data, test_labels, *, batch_size=5, epochs=1000,
                  patch_size=[50, 50, 10], save_level=100, net_save_path='../logs/acts_network.tf'):
    """
    Trains the network with the given batchsize and for a certain amount of epochs.

    :param train_data: the data to be trained on
    :param train_labels: the labels for the data
    :param test_data: the data for checking the performance of the network
    :param test_labels: the labels for checking the performance
    :param batch_size: the batch_size (default is 20)
    :param epochs: the number of epochs (default is 100)
    :param patch_size: the patch_size of th lung scan
    :param save_level: defines at how many epochs the performance is saved
    :param net_save_path: the path the trained network is saved to, is also used for replaying
    :return: epochs and losses depending on the save_level
    """

    train_data_ph = tf.placeholder(tf.float32, [None, patch_size[0], patch_size[1], patch_size[2]])
    train_labels_ph = tf.placeholder(tf.bool, [None])

    loss, optimizer, target, network_output, accuracy, sum_train_loss, sum_test_loss, sum_train_acc, sum_test_acc = network_model(train_data_ph, train_labels_ph)

    log_path = os.path.join(net_save_path, 'acts_' + datetime.now().isoformat())

    # variables for plotting
    losses = []
    epochs_val = []

    global_step = 0

    with tf.Session() as sess:

        writer = tf.summary.FileWriter(log_path, sess.graph)
        writer.add_graph(sess.graph)
        saver = tf.train.Saver()

        # initialize the variables
        sess.run(tf.global_variables_initializer())
        for i in range(1, epochs + 1):

            for j in range(1, len(train_data)//batch_size):
                # build a batch
                batch = np.random.permutation(len(train_data))[0:batch_size]
                batch_scans, batch_labels = train_data[batch], train_labels[batch]
                sess.run([optimizer, target, network_output],
                         {train_data_ph: batch_scans, train_labels_ph: batch_labels})
                global_step = global_step + 1

            # logging important information out for tensorboard
            if i % save_level == 0:
                store_values(sess, train_data, train_labels, test_data, test_labels,
                             batch_size, writer, saver, log_path, global_step,
                             sum_train_loss, sum_test_loss, sum_train_acc, sum_test_acc,
                             train_data_ph, train_labels_ph, epochs_val, losses, accuracy)

        writer.close()

    return epochs_val, losses


def store_values(sess, train_data, train_labels, test_data, test_labels,
                 batch_size, writer, saver, log_path, global_step,
                 sum_train_loss, sum_test_loss, sum_train_acc, sum_test_acc,
                 train_data_ph, train_labels_ph, epochs_val, losses, accuracy):
    """
    Saves metrics for this specific network and the session.
    """
    # calculate accuracy on one batch
    logger = logging.getLogger()

    batch_train = np.random.permutation(len(train_data))[0:batch_size]
    batch_scans_train, batch_labels_train = train_data[batch_train], train_labels[batch_train]

    train_acc, train_loss, train_acc_val = sess.run([sum_train_acc, sum_train_loss, accuracy],
                                                    {train_data_ph: batch_scans_train, train_labels_ph: batch_labels_train})

    batch_test = np.random.permutation(len(test_data))[0:batch_size]
    batch_scans_test, batch_labels_test = test_data[batch_test], test_labels[batch_test]

    test_acc, test_loss, test_acc_val = sess.run([sum_test_acc, sum_test_loss, accuracy],
                                                 {train_data_ph: batch_scans_test, train_labels_ph: batch_labels_test})

    epochs_val.append(global_step)
    losses.append(train_acc)

    logger.info('Step: %s, Acc Train: %s, Acc Test: %s', global_step, train_acc_val, test_acc_val)
    writer.add_summary(train_acc, global_step)
    writer.add_summary(train_loss, global_step)
    writer.add_summary(test_acc, global_step)
    writer.add_summary(test_loss, global_step)
    saver.save(sess, log_path)

if __name__ == "__main__":
    # Todo: think about some tests or func checks
    print('Nothing here yet')
