import tensorflow as tf
import numpy as np
import logging


def network_model(data, labels, *, patch_size=[20, 20, 3]):
    """
    The graph for the tensorflow model that is currently used.

    :param data: the scan cubes as a list
    :param labels: the labels for the lung scan cubes (1 for nodule, 0 for healthy)
    :param patch_size: the patch_size of th lung scan
    :return: the loss of the network
    """
    input_layer = tf.reshape(data, [-1, patch_size[0], patch_size[1], patch_size[2], 1])

    # Convolutional layers with pooling
    conv1 = tf.layers.conv3d(
        inputs=input_layer,
        filters=40,
        kernel_size=[3, 3, 3],
        padding="same")

    pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[2, 2, 1], strides=1)

    filter_num = 20

    conv2 = tf.layers.conv3d(
        inputs=pool1,
        filters=filter_num,
        kernel_size=[3, 3, 3],
        padding="same")
    pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 1], strides=1)

    pool2_flat = tf.reshape(pool2, [-1, (patch_size[0]-2)*(patch_size[1]-2)*patch_size[2]*filter_num])

    # Fully conected Layers with dropout
    dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4)

    dense2 = tf.layers.dense(inputs=dropout1, units=500, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.4)

    nodule_class = tf.layers.dense(inputs=dropout2, units=2)

    # Training labels and loss
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    total_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=nodule_class)
    optimizer = tf.train.AdamOptimizer().minimize(total_loss)

    return total_loss, optimizer, onehot_labels, nodule_class


def simple_network_model(data, labels, *, patch_size=[20, 20, 3]):
    """
    The graph for the tensorflow model that is currently used.

    :param data: the scan cubes as a list
    :param labels: the labels for the lung scan cubes (1 for nodule, 0 for healthy)
    :param patch_size: the patch_size of th lung scan
    :return: the loss of the network
    """
    input_layer = tf.reshape(data, [-1, patch_size[0], patch_size[1], patch_size[2], 1])

    # Convolutional layers with pooling
    conv1 = tf.layers.conv3d(
        inputs=input_layer,
        filters=40,
        kernel_size=[3, 3, 3],
        padding="same")

    pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[2, 2, 1], strides=1)

    #filter_num = 20

    #conv2 = tf.layers.conv3d(
    #    inputs=pool1,
    #    filters=filter_num,
    #    kernel_size=[3, 3, 3],
    #    padding="same")
    #pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 1], strides=1)

    pool2_flat = tf.reshape(pool1, [-1, (patch_size[0]-1)*(patch_size[1]-1)*patch_size[2]*40])

    # Fully conected Layers with dropout
    dense1 = tf.layers.dense(inputs=pool2_flat, units=20, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4)

    #dense2 = tf.layers.dense(inputs=dropout1, units=500, activation=tf.nn.relu)
    #dropout2 = tf.layers.dropout(inputs=dense2, rate=0.4)

    nodule_class = tf.layers.dense(inputs=dropout1, units=2)

    # Training labels and loss
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    total_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=nodule_class)
    optimizer = tf.train.AdamOptimizer().minimize(total_loss)

    return total_loss, optimizer, onehot_labels, nodule_class


def train_network(train_data, train_labels, *, batch_size=5, epochs=1000,
                  patch_size=[20, 20, 3], save_level=100, net_save_path='acts_network.tf'):
    """
    Trains the network with the given batchsize and for a certain amount of epochs.

    :param train_data: the data to be trained on
    :param train_labels: the labels for the data
    :param batch_size: the batch_size (default is 20)
    :param epochs: the number of epochs (default is 100)
    :param patch_size: the patch_size of th lung scan
    :param save_level: defines at how many epochs the performance is saved
    :param net_save_path: the path the trained network is saved to, is also used for replaying
    :return: epochs and losses depending on the save_level
    """
    logger = logging.getLogger()

    train_data_ph = tf.placeholder(tf.float32, [batch_size, patch_size[0], patch_size[1], patch_size[2]])
    train_labels_ph = tf.placeholder(tf.bool, [batch_size])

    loss, optimizer, target, network_output = simple_network_model(train_data_ph, train_labels_ph)

    # variables for plotting
    losses = []
    epochs_val = []

    with tf.Session() as sess:
        saver = tf.train.Saver()

        if True:
            # initialize the variables
            sess.run(tf.global_variables_initializer())
            for i in range(1, epochs + 1):
                # build a batch
                batch = np.random.permutation(len(train_data))[0:batch_size]
                batch_scans, batch_labels = train_data[batch], train_labels[batch]
                _, realLabel, netThought = sess.run([optimizer, target, network_output], {train_data_ph: batch_scans, train_labels_ph: batch_labels})

                if i % save_level == 0:
                    loss_val = sess.run(loss, {train_data_ph: batch_scans, train_labels_ph: batch_labels})
                    epochs_val.append(i)
                    losses.append(loss_val)
                    logger.info('Epoch: %s, Loss: %s', i, loss_val)
                    saver.save(sess, 'acts_network.tf')
        else:
            saver.restore(sess, 'acts_network.tf')
            logger.info('Evaluate network performance on all data')
            sess.run(nodule_class, {train_data_ph: train_data, train_labels_ph: train_labels})

        sess.close()

    return epochs_val, losses

if __name__ == "__main__":
    # Todo: think about some tests or func checks
    print('Nothing here yet')
