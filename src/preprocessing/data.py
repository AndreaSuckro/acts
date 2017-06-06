import numpy as np
import os
import time
import random
import logging
from cvloop import cvloop


def get_train_data(data_dir, *, patch_number=100, tumor_rate=0.5):
    """
    Reads all training data as specified in the training data.

    :param data_dir: the directory that contains a folder for the train data
    :param patch_number: the number of patches that should be retrieved
    :param tumor_rate: the fraction of tumors that should be contained in the patches
    :return: data with nodules and without
    """
    return get_patches(os.path.join(data_dir, 'processed', 'train'),
                       patch_number=patch_number,
                       tumor_rate=tumor_rate)


def get_test_data(data_dir, *, patch_number=100, tumor_rate=0.5):
    """
    Returns all data under the test directory.

    :param data_dir: the directory that contains a folder for the test data
    :param patch_number: the number of patches that should be retrieved
    :param tumor_rate: the fraction of tumors that should be contained in the patches
    :return: data with nodules and without
    """
    return get_patches(os.path.join(data_dir, 'processed', 'test'),
                       patch_number=patch_number,
                       tumor_rate=tumor_rate)


def get_patches(path, *, patch_number=100, tumor_rate=0.5):
    """
    Reads all CT-Scans from a folder with several patients in it.

    :param path: the path to the folder with the patient files
    :param patch_number: the number of patches per patient that should be retrieved
    :param patch_size: the 3d extend of the scan patches
    :param tumor_rate: the fraction of tumors that should be contained in the patches
    :return: datacubes and their labels
    """
    logger = logging.getLogger()

    count = 0
    start_time = time.time()

    data_nod_all = get_rand_samples(os.path.join(path, 'nodules'), int(patch_number*tumor_rate))
    data_health_all = get_rand_samples(os.path.join(path, 'health'), patch_number - int(patch_number*tumor_rate))

    logger.info('Read ct scan data from %s patients in %d seconds.', count, time.time() - start_time)
    data = data_nod_all + data_health_all
    labels_nod = [1] * len(data_nod_all)
    labels_health = [0] * len(data_health_all)
    labels = labels_nod + labels_health

    combined = list(zip(data, labels))
    random.shuffle(combined)
    logger.info('Successfully read in %s lung patches.', len(data))

    return zip(*combined)


def get_rand_samples(path, number):
    """
    Reads n different sample numpy arrays from the specified folder.
    
    :param path: the path thi the patches
    :param number: the number of patches that should be read
    :return: a list of patches
    """
    data = []

    files = random.sample(os.listdir(path), number)
    for file in files:
        data.append(np.load(os.path.join(path, file)))

    return data


if __name__ == "__main__":

    train_data, labels = get_train_data('../../data/', patch_number=10, tumor_rate=0.5)

    print('Labels: {}'.format(str(labels)))

    class Data:
        def __init__(self, scans, lab):
            self.scans = scans
            self.labels = lab
            self.i = 0

        def read(self):
            time.sleep(0.3)  # delays for 0.3 seconds
            self.i = self.i + 1 if self.i < self.scans.shape[2] - 1 else 0
            img = self.scans[:, :, self.i]
            return True, img


    # plot a tumor patch
    cvloop(Data(train_data[labels.index(1)], labels[labels.index(1)]))
