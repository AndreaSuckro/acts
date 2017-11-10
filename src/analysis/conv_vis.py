import time
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from read_network import get_conv_kernels, get_activations, inspect_variables, load_graph


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
    saver = load_graph()
    with tf.Session() as sess:
        saver.restore(sess, '../../data/networks/huang1/acts_2017-09-19T12-37_Huang_no_scaling_50x50')

        kernels = get_conv_kernels(sess)
        input_data = np.random.rand(1,50,50,5)
        activations = get_activations(sess, kernels[0], input_data)

        plot_pca(activations)
        plot_nn_filter(activations)

        plt.show()
