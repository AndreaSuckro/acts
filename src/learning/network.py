from preprocessing.data import get_train_data, get_test_data
from visualization.network_visualiser import plot_loss, plot_sample
import tensorflow as tf
import numpy as np

BATCH_SIZE = 20

PATCH_SIZE = [20, 20, 3]


def network_model(data, labels):
    """
    The graph for the tensorflow model that is currently used.
    
    :param data: the scan cubes as a list
    :param labels: the labels for the lung scan cubes (1 for nodule, 0 for healthy)
    :return: the loss of the network
    """
    input_layer = tf.reshape(data, [-1, PATCH_SIZE[0], PATCH_SIZE[1], PATCH_SIZE[2], 1])

    conv1 = tf.layers.conv3d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5, 3],
        padding="same")

    pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[2, 2, 1], strides=1)

    conv2 = tf.layers.conv3d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5, 3],
        padding="same")
    pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 1], strides=1)

    pool2_flat = tf.reshape(pool2, [-1, (PATCH_SIZE[0]-2)*(PATCH_SIZE[1]-2)*PATCH_SIZE[2]*64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)

    nodule_class = tf.layers.dense(inputs=dropout, units=2)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    total_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=nodule_class)
    #total_loss = tf.losses.mean_squared_error(labels=onehot_labels, predictions=nodule_class)

    optimizer = tf.train.AdamOptimizer().minimize(total_loss)

    return total_loss, optimizer, onehot_labels, nodule_class


def train_network(train_data, train_labels, batch_size=5, epochs=10):
    """
    Trains the network with the given batchsize and for a certain amount of epochs.
    
    :param train_data: the data to be trained on 
    :param train_labels: the labels for the data
    :param batch_size: the batch_size (default is 20)
    :param epochs: the number of epochs (default is 100)
    :return: 
    """

    train_data_ph = tf.placeholder(tf.float32, [batch_size, PATCH_SIZE[0], PATCH_SIZE[1], PATCH_SIZE[2]])
    train_labels_ph = tf.placeholder(tf.bool, [batch_size])

    loss, optimizer, target, network_output = network_model(train_data_ph, train_labels_ph)

    # variables for plotting
    losses = []
    epochs_val = []

    with tf.Session() as sess:
        # initialize the variables
        sess.run(tf.global_variables_initializer())
        for i in range(1, epochs + 1):
            # build a batch
            batch = np.random.permutation(len(train_data))[0:batch_size]
            batch_scans, batch_labels = train_data[batch], train_labels[batch]
            _, realLabel, netThought = sess.run([optimizer, target, network_output], {train_data_ph: batch_scans, train_labels_ph: batch_labels})

            if i % 1 == 0:
                loss_val = sess.run(loss, {train_data_ph: batch_scans, train_labels_ph: batch_labels})
                epochs_val.append(i)
                losses.append(loss_val)
                print(f'Epoch: {i} with loss {loss_val}')
                #print(f'Real label was {realLabel} \n Network Output {netThought}')
        sess.close()

    return epochs_val, losses

if __name__ == "__main__":
    # get the data
    train_data_raw, train_labels_raw = get_train_data(patch_number=20, patch_size=PATCH_SIZE)

    train_data_raw = np.asarray(train_data_raw)
    train_labels_raw = np.asarray(train_labels_raw)

    plot_sample(train_data_raw, train_labels_raw)

    epochs_val, losses = train_network(train_data_raw, train_labels_raw)

    #saveResults()
    plot_loss(epochs_val, losses)


