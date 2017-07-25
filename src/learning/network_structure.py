import tensorflow as tf


def network_model(data, labels, *, patch_size=[50, 50, 3]):
    """
    The graph for the tensorflow model that is currently used.

    :param data: the scan cubes as a list
    :param labels: the labels for the lung scan cubes (1 for nodule, 0 for healthy)
    :param patch_size: the patch_size of th lung scan
    :return: the loss of the network
    """
    input_layer = tf.reshape(data, [-1, patch_size[0], patch_size[1], patch_size[2], 1])

    #########################################################
    # Convolutional layers with pooling
    filter_num1 = 25
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

    pool2_flat = tf.reshape(pool2, [-1, (patch_size[0] - 3) * (patch_size[1] - 3)
                                    * (patch_size[2] - 2) * filter_num2])

    #########################################################
    # Fully connected Layer with dropout
    dense1 = tf.layers.dense(inputs=pool2_flat, units=50,
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

    return total_loss, optimizer, onehot_labels, nodule_class, accuracy, sum_train_loss, sum_test_loss, \
           sum_train_acc, sum_test_acc
