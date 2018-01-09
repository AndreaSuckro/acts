import time
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from read_network import get_conv_kernels, get_activations, inspect_variables, load_graph
from vispy import app,  visuals, scene, io

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from preprocessing.data import get_test_data


def plot_nn_filter(activations, columns=5, figsize=(15, 15), title='All the Activations'):
    """
    Plots the kernel of the conv layer hopefully in a beautiful way.
    """
    print(f'Activations come in shape {activations.shape}')
    filters = activations.shape[4]
    n_columns = columns
    n_rows = math.ceil(filters / n_columns)
    print(f'{n_columns}x{n_rows}')
    print(f'Number of filters is {filters}, displaying in {n_columns}x{n_rows}')
    fig = plt.figure(title, figsize=figsize)
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i+1))
        plt.imshow(np.squeeze(activations[:,:,:,1,i]), interpolation="nearest", cmap="gray")
    fig.tight_layout()
    fig.show()


def plot_pca(activations, title='PCA of kernel activation of the first layer'):
    """
    plot filters with pca to see if there is a difference between them
    """
    activations_m = np.squeeze(activations)
    print(f'Activations Shape after squeeze= {activations_m.shape}')
    filters = activations.shape[3]
    activations_m = np.rollaxis(activations_m, -1)
    print(f'Activations Shape after rolling= {activations_m.shape}')

    activations_m = np.reshape(activations_m, (40, 12500))
    print(f'Activations Shape after shaping= {activations_m.shape}')

    pca = sklearnPCA(n_components=2)  # keep 2 components
    pca.fit(np.array(activations_m))
    transformed = pca.transform(activations_m)

    print(f'Explained variance = {pca.explained_variance_ratio_}')
    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.add_subplot(111)

    for i in range(filters):
        ax.scatter(transformed[i][0], transformed[i][1], label='Filter '+str(i))
    fig.show()


def draw_kernel(activations, filter_num):

    input_data = np.squeeze(activations[0,:,:,:,filter_num])

    print(f'Activation was in form {activations.shape}')
    print(f'Input was in form {input_data.shape}')

    canvas = scene.SceneCanvas(title='Filter '+str(filter_num),keys='interactive', show=True)

    view = canvas.central_widget.add_view()
    cam = scene.cameras.FlyCamera(parent=view.scene, fov=60.0, name='FlyCam')
    view.camera = cam

    volume = scene.visuals.Volume(vol=input_data, cmap='grays', parent=view.scene)
    axis = scene.visuals.XYZAxis(parent=view.scene)
    app.run()


if __name__ == "__main__":
    data_root = ''
    net_name = 'acts_2017-11-21T10-04_dropout_05_more_kernel_and_batch'
    saver = tf.train.import_meta_graph('../../data/networks/final/'+net_name+'.meta')

    test_data_raw, test_labels_raw = get_test_data('/net/store/cv/projects/datasets/image/pub/LIDC-IDRI/', patch_number=500)

    plot3D = True # should the output be just one random nodule kernel in 3D?
    layer_no = 1  # which layer number should be investigated?

    with tf.Session() as sess:
        saver.restore(sess, '../../data/networks/final/'+net_name)

        kernels = get_conv_kernels(sess)
        # Step 1: get a nodule patch
        print(len(test_data_raw))
        nod_patch = []
        health_patch = []
        for idx, label in enumerate(test_labels_raw):
            patch = np.array(test_data_raw[idx], copy=True)
            if label > 0:
                nod_patch.append(patch)
            else:
                health_patch.append(patch)

        first_layer_nodules = get_activations(sess, kernels[layer_no], nod_patch)
        first_layer_health = get_activations(sess, kernels[layer_no], health_patch)

        # mean activation over first dimension for batch
        mean_act_nod = np.mean(first_layer_nodules, axis=0)
        mean_act_nod = mean_act_nod[np.newaxis, :]

        mean_act_health = np.mean(first_layer_health, axis=0)
        mean_act_health = mean_act_health[np.newaxis, :]

        if not plot3D:
            plot_nn_filter(mean_act_nod, title='Nodule Activation')
            plot_nn_filter(mean_act_health, title='Health Activation')
            plt.show()
        else:
            filter_num = np.random.randint(0, mean_act_nod.shape[4])
            draw_kernel(mean_act_nod, filter_num)
