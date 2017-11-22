import time
import numpy as np
import itertools
import tensorflow as tf
from read_network import get_conv_kernels, get_activations, load_graph, inspect_variables

# I am sorry python but you let me no choice ...
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from preprocessing.data import get_test_data
from optparse import OptionParser


def get_commandline_args():
    """
    Creates the command line arguments and returns their values.
    """
    usage = "usage: python3 %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("-d", "--data", dest="data_dir", help="path to the data directory for testing")
    parser.add_option("-n", "--net_save_path", dest="net_save_path",
                      default='?', help="Full path to the stored network (without .meta)")

    return parser.parse_args()


if __name__ == "__main__":

    (option, args) = get_commandline_args()
    test_data = option.data_dir

    # ../../data/networks/huang1/acts_2017-09-19T12-37_Huang_no_scaling_50x50

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(option.net_save_path + '.meta')
        saver.restore(sess, option.net_save_path)

        graph = tf.get_default_graph()
        _, placeholders = inspect_variables(sess, verbose=False)

        input_ph = graph.get_tensor_by_name(placeholders[0].name+":0")
        label_ph = graph.get_tensor_by_name(placeholders[1].name+":0")
        phase_ph = graph.get_tensor_by_name(placeholders[2].name+":0")

        feed_dict = {input_ph:np.random.rand(1,50,50,5), label_ph:[False], phase_ph:False}

        pred = graph.get_tensor_by_name("ArgMax_1:0")
        accuracy = graph.get_tensor_by_name("Mean:0")

        kernels = get_conv_kernels(sess)
        kern_num = kernels[2].shape[4]
        print(f'Number of Kernels {kern_num}')
        print()

        print('And the result is.....')
        print(sess.run([pred, accuracy], feed_dict))
