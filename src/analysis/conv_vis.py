import time
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def get_activations(sess, kernel, input_data, phase, data):
    """
    Runs the data through the kernel and returns the output in a plot.
    """
    activations = sess.run("conv1/conv/convolution:0",feed_dict={input_data.name+":0": data, phase.name+":0": 1})
    print(f'Activations Shape= {activations.shape}')
    return activations


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


def plot_nn_filter(activations, columns=8, figsize=(20,20)):
    """
    Plots the kernel of the conv layer hopefully in a beautiful way.
    """
    print(f'Activations come in shape {activations.shape}')
    filters = activations.shape[4]
    plt.figure(1, figsize=figsize)
    n_columns = columns
    n_rows = math.ceil(filters / n_columns) + 1
    print(f'Number of filters is {filters}, displaying in {n_columns}x{n_rows}')
    fig = plt.figure()
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i+1))
        plt.imshow(np.squeeze(activations[:,:,:,1,i]), interpolation="nearest", cmap="gray")
    fig.show()


def plot_pca(activations):
    """
    plot filters with pca to see if there is a difference between them
    """
    activations = np.squeeze(activations)
    print(f'Activations Shape after squeeze= {activations.shape}')
    filters = activations.shape[3]
    activations = np.rollaxis(activations, -1)
    print(f'Activations Shape after rolling= {activations.shape}')

    activations = np.reshape(activations,(32,12500))
    print(f'Activations Shape after shaping= {activations.shape}')

    pca = sklearnPCA(n_components=2) # keep 2 components
    pca.fit(np.array(activations))
    transformed = pca.transform(activations)

    print(f'Explained variance = {pca.explained_variance_ratio_}')
    fig = plt.figure()
    fig.suptitle('PCA of kernel activation of the first layer')
    ax = fig.add_subplot(111)

    for i in range(filters):
        ax.scatter(transformed[i][0], transformed[i][1], label='Filter '+str(i))
    fig.show()

if __name__ == "__main__":
    # First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('../../data/networks/huang1/acts_2017-09-19T12-37_Huang_no_scaling_50x50.meta')
    with tf.Session() as sess:
        saver.restore(sess, '../../data/networks/huang1/acts_2017-09-19T12-37_Huang_no_scaling_50x50')

        _, placeholders = inspect_variables(sess)
        kernels = get_conv_kernels(sess)

        input_ph = placeholders[0]
        phase_ph = [k for k in sess.graph.get_operations() if "phase" in k.name][0]
        #print(phase_ph)
        activations = get_activations(sess, kernels[0], input_ph, phase_ph, np.random.rand(1,50,50,5))

        plot_pca(activations)
        plot_nn_filter(activations)
        # to ensure that everything only closes when it's done
        plt.show()
