from preprocessing.data import get_train_data
from visualization.network_visualiser import plot_loss
import tensorflow as tf
import numpy as np

BATCH_SIZE = 20


def network_model(data, labels):
    """
    The graph for the tensorflow model that is currently used.
    
    :param data: the scan cubes as a list
    :param labels: the labels for the lung scan cubes (1 for nodule, 0 for healthy)
    :return: the loss of the network
    """
    # Input Layer
    input_layer = tf.reshape(data, [-1, 15, 15, 3, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv3d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5, 3],
        padding="same")

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[2, 2, 1], strides=1)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv3d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5, 3],
        padding="same")
    pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 1], strides=1)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 13*13*3*64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)

    # Logits Layer
    nodule_class = tf.layers.dense(inputs=dropout, units=2)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=nodule_class)
    total_loss = tf.losses.mean_squared_error(labels=onehot_labels, predictions=nodule_class)
    optimizer = tf.train.AdamOptimizer().minimize(total_loss)

    return total_loss, optimizer

if __name__ == "__main__":
    # get the data
    train_data_raw, train_labels_raw = get_train_data(patch_number=200)

    train_data_raw = np.asarray(train_data_raw)
    train_labels_raw = np.asarray(train_labels_raw)

    batch_size = 20
    epochs = 100

    train_data = tf.placeholder(tf.float32, [batch_size, 15, 15, 3])
    train_labels = tf.placeholder(tf.bool, [batch_size])

    loss, optimizer = network_model(train_data, train_labels)

    # variables for plotting
    losses = []
    epochs_val = []

    with tf.Session() as sess:
        # initialize the variables
        sess.run(tf.global_variables_initializer())
        for i in range(1, epochs + 1):
            # build a batch
            batch = np.random.permutation(len(train_data_raw))[0:batch_size]
            batch_scans, batch_labels = train_data_raw[batch], train_labels_raw[batch]
            sess.run(optimizer, {train_data: batch_scans, train_labels: batch_labels})
            if i % 5 == 0:
                loss_val = sess.run(loss, {train_data: batch_scans, train_labels: batch_labels})
                epochs_val.append(i)
                losses.append(loss_val)
                print(f'Epoch: {i} with loss {loss_val}')
        sess.close()

    #saveResults()
    plot_loss(epochs_val, losses)


