import tensorflow as tf
import time


def visualize_kernel(kernel, grid_Y, grid_X, pad=1):
    """
    Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    """

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel-x_min)/(x_max-x_min)

    # pad between the kernels
    padded_kernels = tf.pad(kernel1, tf.constant([[pad,pad], [pad,pad], [0,0], [0,0]]), mode='CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(padded_kernels, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype=tf.uint8)


if __name__ == "__main__":
    sess = tf.Session()
    # First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('../../data/networks/huang1/acts_2017-09-19T12-37_Huang_no_scaling_50x50.meta')
    saver.restore(sess, '../../data/networks/huang1/acts_2017-09-19T12-37_Huang_no_scaling_50x50')

    for i in range(10):
        test = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[i]
        print(test)

    time.sleep(1)
    print('Try to get values')
    with tf.variable_scope('conv1', reuse=None):
        tf.get_variable_scope().reuse_variables()
        print(tf.get_variable('conv/kernel:0'))

    # Visualize conv1 kernels
    #with tf.variable_scope('conv1'):
    #    tf.get_variable_scope().reuse_variables()
    #    kernels = tf.get_variable('conv/kernel')
    #    grid = visualize_kernel(kernels)
    #    tf.image.summary('conv1/kernels', grid, max_outputs=1)