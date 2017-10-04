import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import math


def get_activations(sess, kernel, input_data, phase, data):
    """
    Runs the data through the kernel and returns the output in a plot.
    """
    activations = sess.run("conv2/conv/convolution:0",feed_dict={input_data.name+":0": data, phase.name+":0": 1})
    print(f'Activations = {activations}')
    plot_nn_filter(activations)


def plot_nn_filter(units, columns=8, figsize=(20,20)):
    """
    Plots the kernel of the conv layer hopefully in a beautiful way.
    """
    print(f'Activations come in shape {units.shape}')
    filters = units.shape[4]
    plt.figure(1, figsize=figsize)
    n_columns = columns
    n_rows = math.ceil(filters / n_columns) + 1
    print(f'Number of filters is {filters}, displaying in {n_columns}x{n_rows}')

    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i+1))
        #print(units[:,:,:,:,1])
        plt.imshow(np.squeeze(units[:,:,:,:,i]), interpolation="nearest", cmap="gray")
    plt.show()


def inspect_variables(sess, full=False):
    """
    Inspects the variables stored in the session and prints out
    information for the trainable variables as well as the placeholders.
    """
    print('#####List of all TRAINABLE variables#####\n')
    kernels = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print(f'Found {len(kernels)} trainable variables in the graph:\n')
    for var in kernels:
        print(var)
        if full:
            print(var.eval())

    print('\n#####List of placeholders#####\n')
    placeholders = [x for x in sess.graph.get_operations() if "Placeholder" in x.name]
    print(f'Found {len(placeholders)} placeholders in the graph:\n')
    for ph in placeholders:
        print(ph.name)
        print(ph.get_attr("shape"))

    return kernels, placeholders


def get_conv_kernels(sess):
    """
    A function to get all the convolutional kernels from a graph.
    """
    kernels = [k for k in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if "conv/kernel" in k.name]
    for k in kernels:
        print(k)
    print(f'Found {len(kernels)} layers with convolution outputs.')
    return kernels


if __name__ == "__main__":
    # First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('../../data/networks/huang1/acts_2017-09-19T12-37_Huang_no_scaling_50x50.meta')
    with tf.Session() as sess:
        saver.restore(sess, '../../data/networks/huang1/acts_2017-09-19T12-37_Huang_no_scaling_50x50')

        _, placeholders = inspect_variables(sess)
        kernels = get_conv_kernels(sess)

        input_ph = placeholders[0]
        phase_ph = [k for k in sess.graph.get_operations() if "phase" in k.name][0]
        print(phase_ph)
        get_activations(sess, kernels[0], input_ph, phase_ph, np.ones((1,50,50,5)))
