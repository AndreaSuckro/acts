import tensorflow as tf
import numpy as np


def conv2d_layer(scope, input, phase, *, num_filters=20, kernel_size=[3, 3],
                 kernel_stride=[1, 1], pool_size=[2, 2], pool_stride=1):
    """
    Creates a 2d convolutional layer with batchnorm and dropout followed by pooling.

    :param scope: the scope for this layer
    :param input: the input tensor to this layer
    :param phase: either test or train
    :param num_filters: number of filter kernels to be used
    :param kernel_size: the size of the filter kernels
    :param kernel_stride: the stride of the kernel
    :param pool_size: the pooling size
    :param pool_stride: the stride of the pooling kernel
    :return: the activation of the layer
    """
    with tf.variable_scope(scope) as scope:
        conv = tf.layers.conv2d(inputs=input,
                                filters=num_filters,
                                kernel_size=kernel_size,
                                strides=kernel_stride,
                                padding="same",
                                name="conv")
        bn = tf.layers.batch_normalization(conv, center=True, scale=True,
                                           training=phase)
        #dropout = tf.layers.dropout(inputs=bn, rate=0.01, name="dropout", training=phase)
        pool = tf.layers.max_pooling2d(inputs=bn, pool_size=pool_size,
                                       strides=pool_stride, name='pool')
    return pool


def conv3d_layer(scope, input, phase, *, num_filters=20, kernel_size=[5, 5, 3],
                 kernel_stride=[1, 1, 1], pool_size=[2, 2, 2], pool_stride=1):
    """
    Creates a 3d convolutional layer with batchnorm and dropout followed by pooling.

    :param scope: the scope for this layer
    :param input: the input tensor to this layer
    :param phase: either test or train
    :param num_filters: number of filter kernels to be used
    :param kernel_size: the size of the filter
    :param kernel_stride: how the kernel strides over the image
    :param pool_size: the pooling size
    :param pool_stride: the stride of the pooling kernel
    :return: the activation of therescal layer
    """
    with tf.variable_scope(scope):
        conv = tf.layers.conv3d(inputs=input,
                                filters=num_filters,
                                kernel_size=kernel_size,
                                strides=kernel_stride,
                                padding="same",
                                name="conv")
        bn = tf.layers.batch_normalization(conv, center=True, scale=True,
                                           training=phase)
        #dropout = tf.layers.dropout(inputs=bn, rate=0.01, name="dropout", training=phase)
        pool = tf.layers.max_pooling3d(inputs=bn, pool_size=pool_size,
                                       strides=pool_stride, name='pool')
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
        #dropout = tf.layers.dropout(inputs=bnd, rate=0.5, name="dropout",
        #                            training=phase)

    return bnd


def augment_data(input_data):
    """
    Augments the data with random flip left right and up down.
    """
    data = tf.image.random_flip_left_right(input_data)
    data = tf.image.random_flip_up_down(data)
    return data


def input_summary(aug_data, labels):
    """
    Generates a patch image for a healthy and annormal_net
    unhealthy patch.
    """
    nodule_idx = tf.gather(tf.where(tf.equal(labels, True)), 0)
    health_idx = tf.gather(tf.where(tf.equal(labels, False)), 0)

    slice_nodule = tf.slice(tf.gather(aug_data, nodule_idx), [0,0,0,0], [-1,-1,-1,1], name = 'nodule_sum')
    slice_health = tf.slice(tf.gather(aug_data, health_idx), [0,0,0,0], [-1,-1,-1,1], name = 'health_sum')

    sum_health_img = tf.summary.image('health_patch', slice_health, max_outputs=1)
    sum_nodule_img = tf.summary.image('nodule_patch', slice_nodule, max_outputs=1)

    return sum_health_img, sum_nodule_img


def network_model(data, labels, scale_size, *, patch_size=(20, 20, 5)):
    """
    The graph for the tensorflow model that is currently used.

    :param data: the scan cubes as a list
    :param labels: the labels for the lung scan cubes (1 for nodule, 0 for healthy)
    :param patch_size: the patch_size of the lung scan
    :return: the loss of the network
    """
    phase = tf.placeholder(tf.bool, name='phase')
    scaled_data = tf.image.resize_images(data, scale_size)
    aug_data = tf.map_fn(augment_data, scaled_data)
    sum_health_img, sum_nodule_img = input_summary(aug_data, labels)

    ########################################################
    # Layers
    input_layer = tf.reshape(aug_data, [-1, scale_size[0], scale_size[1], patch_size[2], 1])

    #########################################################
    # Convolutional layers

    conv1 = conv3d_layer('conv1', input_layer, phase, num_filters=32,
                         kernel_size=3, pool_size=2, pool_stride=1)

    conv2 = conv3d_layer('conv2', conv1, phase, num_filters=16,
                         kernel_size=3, pool_size=2, pool_stride=1)

    conv3 = conv3d_layer('conv3', conv2, phase, num_filters=16,
                          kernel_size=3, pool_size=2, pool_stride=1)

    pool3_flat = tf.contrib.layers.flatten(conv3)

    #########################################################
    # Fully connected Layer with dropout

    dens1 = dense_layer('dense1', pool3_flat, phase, num_neurons=64, activation_fun=tf.nn.relu)
    dens2 = dense_layer('dense2', dens1, phase, num_neurons=64, activation_fun=tf.nn.relu)

    nodule_class = tf.layers.dense(inputs=dens2, units=2, name="class")
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
           sum_train_acc, sum_test_acc, phase, sum_health_img, sum_nodule_img
