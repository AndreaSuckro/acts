import tensorflow as tf

if __name__ == "__main__":
    sess = tf.Session()
    # First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('../../data/networks/huang1/')
    saver.restore(sess, tf.train.latest_checkpoint('./'))