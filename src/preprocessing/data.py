import numpy as np
import os
import time
import random
import logging
import re


def get_data(data_dir, folder, *, patch_number=100, tumor_rate=0.5):
    """
    Returns lung patches and labels for the specified folder.

    :param data_dir: the directory that contains all data_dir
    :param folder: can be either 'train' or 'test'
    :param patch_number: the number of patches to be returned
    :param tumor_rate: the fraction of tumors in the returned data
    """
    if folder is 'train':
        return get_train_data(data_dir, patch_number=patch_number, tumor_rate=tumor_rate)
    elif folder is 'validation':
        return get_validation_data(data_dir, patch_number=patch_number, tumor_rate=tumor_rate)


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
    Reads all test data as specified in the training data.

    :param data_dir: the directory that contains a folder for the test data
    :param patch_number: the number of patches that should be retrieved
    :param tumor_rate: the fraction of tumors that should be contained in the patches
    :return: data with nodules and without
    """
    return get_patches(os.path.join(data_dir, 'processed', 'test'),
                       patch_number=patch_number,
                       tumor_rate=tumor_rate)


def get_validation_data(data_dir, *, patch_number=100, tumor_rate=0.5):
    """
    Returns all data under the test directory.

    :param data_dir: the directory that contains a folder for the test data
    :param patch_number: the number of patches that should be retrieved
    :param tumor_rate: the fraction of tumors that should be contained in the patches
    :return: data with nodules and without
    """
    return get_patches(os.path.join(data_dir, 'processed', 'validation'),
                       patch_number=patch_number,
                       tumor_rate=tumor_rate)


def get_data_patient(data_dir, *, patient_num='LIDC-IDRI-0666', dir='train'):
    """
    Searches only in train data for a specific patient number

    :param data_dir: data directory
    :param patient_num: the number of the patient to be read
    :param dir: can be either test,train or validation
    :return: get training data for one patient
    """
    logger = logging.getLogger()
    if data_dir is None:
        logger.error('Data directory was not specified.')
        return

    logger.info('Read train data for patient %s', patient_num)
    full_path = os.path.join(data_dir, 'processed', dir)

    data_nod_all = get_patient_samples(os.path.join(full_path, 'nodules'),
                                       patient_number=patient_num)
    data_health_all = get_patient_samples(os.path.join(full_path, 'health'),
                                          patient_number=patient_num)

    return create_labels(data_nod_all, data_health_all)


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

    start_time = time.time()
    logger.info('Read ct scan data from %s', path)
    logger.info('With patch_number of %d and tumor_rate of %.2f', patch_number, tumor_rate)
    data_nod_all = get_rand_samples(os.path.join(path, 'nodules'), int(patch_number*tumor_rate))
    data_health_all = get_rand_samples(os.path.join(path, 'health'), patch_number - int(patch_number*tumor_rate))

    logger.info('Read ct scan data in %d seconds.', time.time() - start_time)
    logger.info('Successfully read in %s lung patches.', len(data_nod_all) + len(data_health_all))

    return create_labels(data_nod_all, data_health_all)


def create_labels(data_nod, data_health):
    """
    Labels created for the healthy patches and nodule patches.

    :param data_nod: data containing nodules
    :param data_health:  data containing only healthy patches
    :return: data and label list shuffled
    """

    data = np.array(data_nod + data_health)

    labels_nod = [1] * len(data_nod)
    labels_health = [0] * len(data_health)
    labels = np.array(labels_nod + labels_health)

    combined = list(zip(data, labels))
    random.shuffle(combined)

    return zip(*combined)


def get_patient_samples(path, *, patient_number='0666'):
    """
    Get specific data samples for one patient.

    :param path: the folder with the files
    :param number: the number of samples to pick
    :param patient_number: the patient number as a string
    :return: patient_samples as a list
    """
    data = []
    files = [f for f in os.listdir(path) if re.match('.*' + patient_number + '.*', f)]
    for file in files:
        data.append(np.load(os.path.join(path, file)))
    return data


def get_rand_samples(path, number):
    """
    Reads n different sample numpy arrays from the specified folder.

    :param path: the path thi the patches
    :param number: the number of patches that should be read
    :return: a list of patches
    """
    data = []

    files = random.sample(os.listdir(path), int(number))
    for file in files:
        if file.endswith('.npy'):
            data.append(np.load(os.path.join(path, file)))

    return data


if __name__ == "__main__":
    from cvloop import cvloop
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
