import numpy as np
import tensorflow as tf
from vispy import app, visuals, scene, io
from read_network import get_conv_kernels, get_activations, load_graph, inspect_variables


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
    folder_name = 'acts_2017-11-21T10-04_dropout_05_more_kernel_and_batch'
    saver = tf.train.import_meta_graph('../../data/networks/final/' + folder_name + '.meta')

    with tf.Session() as sess:
        saver.restore(sess, '../../data/networks/final/' + folder_name)

        graph = tf.get_default_graph()
        _, placeholders = inspect_variables(sess, verbose=False)
        print(placeholders)
        input_ph = graph.get_tensor_by_name(placeholders[0].name+":0")
        label_ph = graph.get_tensor_by_name(placeholders[1].name+":0")
        phase_ph = graph.get_tensor_by_name(placeholders[2].name+":0")

        feed_dict = {input_ph: np.random.rand(1, 50, 50, 5), label_ph: [False], phase_ph: False}

        op_to_restore = graph.get_tensor_by_name("ArgMax_1:0")

        kernels = get_conv_kernels(sess)
        kern_num = kernels[2].shape[4]
        print(f'Number of Kernels {kern_num}')

        print('And the result is.....')
        print(sess.run(op_to_restore, feed_dict))
