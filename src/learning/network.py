import tensorflow as tf
import numpy as np
import logging
from datetime import datetime
from tools.log import log_args
import os

from learning.network_structure import network_model

PATCH_SIZE = [30, 30, 10]

@log_args
def train_network(train_data, train_labels, validation_data, validation_labels, *, batch_size=5, epochs=1000,
                  patch_size=PATCH_SIZE, save_level=100, net_save_path='../logs/acts_network.tf', test_name='default'):
    """
    Trains the network with the given batchsize and for a certain amount of epochs.

    :param train_data: the data to be trained on
    :param train_labels: the labels for the data
    :param validation_data: the data for checking the performance of the network
    :param validation_labels: the labels for checking the performance
    :param batch_size: the batch_size (default is 20)
    :param epochs: the number of epochs (default is 100)
    :param patch_size: the patch_size of th lung scan
    :param save_level: defines at how many epochs the performance is saved
    :param net_save_path: the path the trained network is saved to, is also used for replaying
    :param test_name: the experiment name that describes the layout or data
    :return: epochs and losses depending on the save_level
    """

    train_data_ph = tf.placeholder(tf.float32, [None, patch_size[0], patch_size[1], patch_size[2]])
    train_labels_ph = tf.placeholder(tf.bool, [None])

    # so many variables!
    loss, optimizer, target, network_output, \
    accuracy, sum_train_loss, sum_validation_loss, \
    sum_train_acc, sum_validation_acc, phase = network_model(train_data_ph, train_labels_ph, patch_size=patch_size)

    log_path = net_save_path

    # variables for plotting
    losses = []
    epochs_val = []

    global_step = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

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
                meta_data = tf.RunMetadata()
                opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if global_step%100000 == 0 else None
                sess.run([optimizer, target, network_output],
                         {train_data_ph: batch_scans, train_labels_ph: batch_labels, phase: 1},
                         run_metadata=meta_data, options=opts)
                if i % save_level + j == 1:
                    writer.add_run_metadata(meta_data, str(global_step))
                global_step = global_step + 1

            # logging important information out for tensorboard
            if i % save_level == 0:
                store_values(sess, train_data, train_labels, validation_data, validation_labels,
                             batch_size, writer, saver, log_path, global_step,
                             sum_train_loss, sum_validation_loss, sum_train_acc, sum_validation_acc,
                             train_data_ph, train_labels_ph, epochs_val, losses, accuracy, phase)

        writer.close()

    return epochs_val, losses


def store_values(sess, train_data, train_labels, validation_data, validation_labels,
                 batch_size, writer, saver, log_path, global_step,
                 sum_train_loss, sum_validation_loss, sum_train_acc, sum_validation_acc,
                 train_data_ph, train_labels_ph, epochs_val, losses, accuracy, phase):
    """
    Saves metrics for this specific network and the session.
    """
    # calculate accuracy on one batch
    logger = logging.getLogger()

    batch_train = np.random.permutation(len(train_data))[0:len(train_data)]
    batch_scans_train, batch_labels_train = train_data[batch_train], train_labels[batch_train]

    train_acc, train_loss, train_acc_val = sess.run([sum_train_acc, sum_train_loss, accuracy],
                                                    {train_data_ph: batch_scans_train,
                                                     train_labels_ph: batch_labels_train, phase: 0})

    # be bold and take whole validation set
    # batch_validation = np.random.permutation(len(validation_data))[0:batch_size]
    batch_validation = np.random.permutation(len(validation_data))[0:len(validation_data)]
    batch_scans_validation, batch_labels_validation = validation_data[batch_validation], \
                                                      validation_labels[batch_validation]

    validation_acc, validation_loss, validation_acc_val = sess.run([sum_validation_acc, sum_validation_loss, accuracy],
                                                                   {train_data_ph: batch_scans_validation,
                                                                    train_labels_ph: batch_labels_validation, phase: 0})

    epochs_val.append(global_step)
    losses.append(train_acc)

    logger.info('Step: %s, Acc Train: %s, Acc validation: %s', global_step, train_acc_val, validation_acc_val)
    writer.add_summary(train_acc, global_step)
    writer.add_summary(train_loss, global_step)
    writer.add_summary(validation_acc, global_step)
    writer.add_summary(validation_loss, global_step)
    saver.save(sess, log_path)

if __name__ == "__main__":
    # Todo: think about some tests or func checks
    print('Nothing here yet')
