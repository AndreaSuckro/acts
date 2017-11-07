
import numpy as np
import tensorflow as tf
from vispy import app, visuals, scene, io
from read_network import get_conv_kernels, get_activations, load_graph


if __name__ == "__main__":
    saver = load_graph()
    input_data = np.random.rand(1,50,50,5)

    with tf.Session() as sess:
        saver.restore(sess, '../../data/networks/huang1/acts_2017-09-19T12-37_Huang_no_scaling_50x50')
        kernels = get_conv_kernels(sess)

        sigma = 0.1
        kernel_num = 3

        for i in range(500):
            activation = get_activations(sess, kernels[0], input_data)[:,:,:,:,kernel_num]
            #update step- change the data accordingly
            derivative = activation - input_data
            input_data = input_data + sigma*derivative

#plot the input data
canvas = scene.SceneCanvas(keys='interactive', show=True)

view = canvas.central_widget.add_view()
view.camera = 'fly'
view.camera.fov = 50
view.camera.distance = 200

print(f'Volume input data {input_data.shape}')

volume = scene.visuals.Volume(vol=np.squeeze(input_data), cmap='viridis', parent=view.scene)
axis = scene.visuals.XYZAxis(parent=view.scene)

app.run()
