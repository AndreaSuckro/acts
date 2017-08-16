import tensorflow as tf


def conv3d_layer(scope, input, phase, *, num_filters=20, kernel_size=[5, 5, 3],
                 kernel_stride=[1, 1, 1], pool_size=[2, 2, 2], pool_stride=1):
    """
    Creates a 3d convolutional layer with batchnorm and dropout followed by pooling.

    :param scope: the scope for this layer
    :param input: the input tensor to this layer
    :param phase: either test or train
    :param num_filters: number of filter kernels to be used
    :param kernel_size: the size of the filter kernels
    :param pool_size: the pooling size
    :param pool_stride: the stride of the pooling kernel
    :return: the activation of the layer
    """
    with tf.variable_scope(scope) as scope:
        conv = tf.layers.conv3d(inputs=input,
                                filters=num_filters,
                                kernel_size=kernel_size,
                                strides=kernel_stride,
                                padding="same",
                                name="conv")
        bn = tf.layers.batch_normalization(conv, center=True, scale=True,
                                           training=phase)
        dropout = tf.layers.dropout(inputs=bn, rate=0.01, name="dropout1", training=phase)
        pool = tf.layers.max_pooling3d(inputs=dropout, pool_size=pool_size,
                                       strides=pool_stride, name='pool1')
    return pool


def dense_layer(scope, input, phase, *, num_neurons=50, activation_fun=tf.nn.relu):
    """
    Creates a dense layer with the given parameters.
    """
    with tf.variable_scope(scope):
        dense = tf.layers.dense(inputs=input, units=num_neurons,
                                 activation=activation_fun, name="dense")
        bnd = tf.layers.batch_normalization(dense, center=True, scale=True,
                                             training=phase)
        dropout = tf.layers.dropout(inputs=bnd, rate=0.1, name="dropout",
                                    training=phase)

    return dropout


def network_model(data, labels, *, patch_size=[40, 40, 1]):
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
    # Convolutional layers

    conv1 = conv3d_layer('conv1', input_layer, phase, num_filters=30,
                         kernel_size=[3, 3, 1], kernel_stride=[2, 2, 1], pool_stride=1)

    conv2 = conv3d_layer('conv2', conv1, phase, num_filters=40,
                         kernel_size=[3, 3, 1], pool_size=[2, 2, 1], pool_stride=1)

    conv3 = conv3d_layer('conv3', conv2, phase, num_filters=50,
                         kernel_size=[3, 3, 1], pool_size=[2, 2, 1], pool_stride=1)

    pool3_flat = tf.contrib.layers.flatten(conv3)

    #########################################################
    # Fully connected Layer with dropout

    dens1 = dense_layer('dense', pool3_flat, phase, num_neurons=100, activation_fun = tf.nn.relu)

    nodule_class = tf.layers.dense(inputs=dens1, units=2, name="class")
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

    all_vars = tf.trainable_variables()
    tf.contrib.layers.summarize_tensors(all_vars)

    return total_loss, optimizer, onehot_labels, nodule_class, accuracy, sum_train_loss, sum_test_loss, \
           sum_train_acc, sum_test_acc, phase
