import tensorflow as tf


GRAPH_PATH = '../../data/networks/huang1/acts_2017-09-19T12-37_Huang_no_scaling_50x50.meta'

def load_graph():
    """
    Loads the Huang network that performed so far.
    """
    saver = tf.train.import_meta_graph(GRAPH_PATH)
    return saver


def get_conv_kernels(sess):
    """
    A function to get all the convolutional kernels from a graph. Those are only
    the filter kernels used by the convolutional layer. The application of those
    kernels can be found in the ../convolution node
    """
    kernels = [k for k in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if "conv/kernel" in k.name]
    print(f'Found {len(kernels)} layers with convolution outputs.')
    return kernels


def get_activations(sess, kernel, data):
    """
    Runs the given data through the convolution that the given kernel belongs
    to and returns the putput activation.
    """
    _, placeholders = inspect_variables(sess, verbose=False)
    input_ph = placeholders[0]
    phase_ph = [k for k in sess.graph.get_operations() if "phase" in k.name][0]
    layer_name = "/".join(kernel.name.split('/')[0:2])+"/"
    activations = sess.run(layer_name+"convolution:0",
                           feed_dict={input_ph.name+":0": data, phase_ph.name+":0": 1})
    return activations


def inspect_variables(sess, full=False, verbose=True):
    """
    Inspects the variables stored in the session and prints out
    information for the trainable variables as well as the placeholders.
    """

    kernels = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    placeholders = [x for x in sess.graph.get_operations() if "Placeholder" in x.name]

    if verbose:
        print('\n#####List of all TRAINABLE variables#####\n')
        print(f'Found {len(kernels)} trainable variables in the graph:\n')
        for var in kernels:
            print(var)
            if full:
                print(var.eval())
        print('\n#####List of placeholders#####\n')
        print(f'Found {len(placeholders)} placeholders in the graph:\n')
        for ph in placeholders:
            print(ph.name)
            print(ph.get_attr("shape"))

    return kernels, placeholders


def save_3d_to_disk(matrix, file_path):
    """
    Saving a 3d matrix to disk since np.savetxt doesn't support
    more than 2 dimensions. The input format should be taken directly
    from the TensorFlow variables that are (batch*X*Y*Z*Filter)
    """
    activations = np.squeeze(matrix)
    filters = activations.shape[3]
    # now bring it to format filterxXxYxZ
    activations = np.rollaxis(activations, -1)
    filter1 = activations[1,:,:,:]
    print(f'Filter 1 shape for saving {filter1.shape}')
    with open(file_path, "w") as f:
        for i in range(filter1.shape[0]):
            for j in range(filter1.shape[1]):
                for k in range(filter1.shape[2]):
                    f.write(f'{i},{j},{k},{filter1[i,j,k]}\n')
    print(f'Wrote file successfully to {file_path}')
