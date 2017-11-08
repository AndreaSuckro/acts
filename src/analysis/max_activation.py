import time
import numpy as np
import itertools
from multiprocessing import Pool
import tensorflow as tf
from vispy import app, visuals, scene, io
from read_network import get_conv_kernels, get_activations, load_graph


def draw_kernel(args):
    k, input_data = args
    canvas = scene.SceneCanvas(title='Filter '+str(k),keys='interactive', show=True)

    view = canvas.central_widget.add_view()
    cam = scene.cameras.FlyCamera(parent=view.scene, fov=60.0, name='FlyCam')
    view.camera = cam

    volume = scene.visuals.Volume(vol=np.squeeze(input_data[k,:,:,:,:]), cmap='grays', parent=view.scene)
    axis = scene.visuals.XYZAxis(parent=view.scene)
    app.run()

if __name__ == "__main__":
    saver = load_graph()

    with tf.Session() as sess:
        saver.restore(sess, '../../data/networks/huang1/acts_2017-09-19T12-37_Huang_no_scaling_50x50')
        kernels = get_conv_kernels(sess)
        kern_num = kernels[0].shape[4]

        input_data = np.random.rand(kern_num,1,50,50,5)

        sigma = 0.1
        kernel_num = 3

        for i in range(500):
            for j in range(kern_num):
                activation = get_activations(sess, kernels[0], input_data[j])[...,j]
                #update step- change the data accordingly
                derivative = activation - input_data[j]
                input_data[j] = input_data[j] + sigma*derivative

#plot the input data
print(input_data.shape)

arguments = itertools.zip_longest(range(5),[],fillvalue=input_data)
with Pool(5) as p:
    p.map(draw_kernel, arguments)
