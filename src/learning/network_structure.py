import tensorflow as tf


def network_model(data, labels, *, patch_size=[50, 50, 10]):
    """
    The graph for the tensorflow model that is currently used.

    :param data: the scan cubes as a list
    :param labels: the labels for the lung scan cubes (1 for nodule, 0 for healthy)
    :param patch_size: the patch_size of th lung scan
    :return: the loss of the network
    """
    phase = tf.placeholder(tf.bool, name='phase')
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
    bn1 = tf.layers.batch_normalization(conv1, center=True, scale=True,
                                        training=phase)
    dpo1 = tf.layers.dropout(inputs=bn1, rate=0.01, name="dropout1")
    pool1 = tf.layers.max_pooling3d(inputs=dpo1, pool_size=[2, 2, 2],
                                    strides=1, name='pool1')

    filter_num2 = 50

    conv2 = tf.layers.conv3d(
        inputs=pool1,
        filters=filter_num2,
        kernel_size=[5, 5, 3],
        padding="same",
        name="conv2")
    bn2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                        training=phase)
    dpo2 = tf.layers.dropout(inputs=bn2, rate=0.01, name="dropout2")
    pool2 = tf.layers.max_pooling3d(inputs=dpo2, pool_size=[3, 3, 2],
                                    strides=1, name="pool2")

    filter_num3 = 20

    conv3 = tf.layers.conv3d(
        inputs=pool2,
        filters=filter_num3,
        kernel_size=[15, 15, 3],
        padding="same",
        name="conv3")
    bn3 = tf.layers.batch_normalization(conv3, center=True, scale=True,
                                        training=phase)
    dpo3 = tf.layers.dropout(inputs=bn3, rate=0.01, name="dropout3")
    pool3 = tf.layers.max_pooling3d(inputs=dpo3, pool_size=[10, 10, 8],
                                    strides=2, name="pool3")

    pool3_flat = tf.contrib.layers.flatten(pool3)

    #########################################################
    # Fully connected Layer with dropout
    dense1 = tf.layers.dense(inputs=pool3_flat, units=50,
                             activation=tf.nn.relu, name="dense1")
    bnd1 = tf.layers.batch_normalization(dense1, center=True, scale=True,
                                         training=phase)
    dropout1 = tf.layers.dropout(inputs=bnd1, rate=0.4, name="dropout")

    nodule_class = tf.layers.dense(inputs=dropout1, units=2, name="class")
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)

    #########################################################
    # Optimizer and Accuracy
    total_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                 logits=nodule_class)
    sum_train_loss = tf.summary.scalar("train/loss", total_loss)
    sum_test_loss = tf.summary.scalar("validation/loss", total_loss)

    # for the moving mean of the batch norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer().minimize(total_loss)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(onehot_labels, 1), tf.argmax(nodule_class, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sum_train_acc = tf.summary.scalar("train/acc", accuracy)
    sum_test_acc = tf.summary.scalar("validation/acc", accuracy)

    return total_loss, optimizer, onehot_labels, nodule_class, accuracy, sum_train_loss, sum_test_loss, \
           sum_train_acc, sum_test_acc, phase
