import tensorflow as tf
import numpy as np
import logging
from datetime import datetime
import os

def network_model(data, labels, *, patch_size=[50, 50, 3]):
    """
    The graph for the tensorflow model that is currently used.

    :param data: the scan cubes as a list
    :param labels: the labels for the lung scan cubes (1 for nodule, 0 for healthy)
    :param patch_size: the patch_size of th lung scan
    :return: the loss of the network
    """
    input_layer = tf.reshape(data, [-1, patch_size[0], patch_size[1], patch_size[2], 1])
    filter_num1 = 25
    # Convolutional layers with pooling
    conv1 = tf.layers.conv3d(
        inputs=input_layer,
        filters=filter_num1,
        kernel_size=[5, 5, 3],
        padding="same",
        name="conv1")
    pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[2, 2, 2],
                                    strides=1, name='pool1')

    filter_num2 = 50

    conv2 = tf.layers.conv3d(
        inputs=pool1,
        filters=filter_num2,
        kernel_size=[5, 5, 3],
        padding="same",
        name="conv2")
    pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[3, 3, 2],
                                    strides=1, name="pool2")

    pool2_flat = tf.reshape(pool2, [-1, (patch_size[0]-3) * (patch_size[1]-3)
                                    * (patch_size[2]-2) * filter_num2])

    # Fully conected Layer with dropout
    dense1 = tf.layers.dense(inputs=pool2_flat, units=30,
                             activation=tf.nn.relu, name="dense1")
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4, name="dropout")

    nodule_class = tf.layers.dense(inputs=dropout1, units=2, name="class")
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)

    # Training labels and loss
    total_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                 logits=nodule_class)
    sum_train_loss = tf.summary.scalar("train/loss", total_loss)
    sum_test_loss = tf.summary.scalar("test/loss", total_loss)

    optimizer = tf.train.AdamOptimizer().minimize(total_loss)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(onehot_labels, 1), tf.argmax(nodule_class, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sum_train_acc = tf.summary.scalar("train/acc", accuracy)
    sum_test_acc = tf.summary.scalar("test/acc", accuracy)

    return total_loss, optimizer, onehot_labels, nodule_class, accuracy, sum_train_loss, sum_test_loss, sum_train_acc, sum_test_acc


def train_network(train_data, train_labels, test_data, test_labels, *, batch_size=5, epochs=1000,
                  patch_size=[50, 50, 3], save_level=100, net_save_path='../logs/acts_network.tf'):
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
    logger = logging.getLogger()

    train_data_ph = tf.placeholder(tf.float32, [None, patch_size[0], patch_size[1], patch_size[2]])
    train_labels_ph = tf.placeholder(tf.bool, [None])

    loss, optimizer, target, network_output, accuracy, sum_train_loss, sum_test_loss, sum_train_acc, sum_test_acc = network_model(train_data_ph, train_labels_ph)

    log_path = os.path.join(net_save_path, 'acts_' + datetime.now().isoformat())
    writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())

    # variables for plotting
    losses = []
    epochs_val = []

    global_step = 0

    with tf.Session() as sess:

        saver = tf.train.Saver()

        if True:
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

        else:
            saver.restore(sess, os.path.append(net_save_path, datetime.now() + '_acts_network.tf'))
            logger.info('Evaluate network performance on all data')
            sess.run(network_output, {train_data_ph: train_data, train_labels_ph: train_labels})

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
