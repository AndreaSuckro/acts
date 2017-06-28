import matplotlib.pyplot as plt
import numpy as np
import preprocessing.create_samples as data
import preprocessing.data as d
from cvloop import cvloop
import os
import time


def plot_patient(path):
    # show case the methods of the data module
    scan_pat = data.read_patient(path)
    annos = data.read_annotation(path, scan_pat)
    array_patient = data.conv2array(scan_pat)
    print(f'Shape of scan data: {array_patient.shape}')

    data_nod, data_health = data.slice_patient(array_patient, annos)
    print(f'Shape of nodules: {len(data_nod)}x{data_nod[0].shape}, '
          f'Shape of health: {len(data_health)}x{data_health[0].shape}')

    # create annotations for frames
    formating = {'shape': 'RECT',
                 'color': '#008000',
                 'line': 2,
                 'size': (20, 20)}

    format_list = []

    for anno in annos:
        t = anno.tolist()
        t.append(formating)
        format_list.append(t)

    class Data:
        def __init__(self, scans):
            self.scans = scans
            self.i = 0

        def read(self):
            time.sleep(0.3)  # delays for 0.3 seconds
            self.i = self.i + 1 if self.i < self.scans.shape[2] - 1 else 0
            img = self.scans[:, :, self.i]
            return True, img

    cvloop(Data(array_patient), annotations=annos, print_info=True)


def plot_data(train_data, train_label):
    """
    Plots a subplot with a random positive and negative case sample.

    :param train_data: training lung ct patch
    :param train_label: the label per patch
    :param number: the number of samples that should be plotted
    :return: a plot with a sample from both classes
    """

    nodule_idxs = [i for i, x in enumerate(train_label) if x == 1]
    health_idxs = [i for i, x in enumerate(train_label) if x == 0]

    g = plt.figure(1)
    plt.suptitle(f'Tumor and non-Tumor Patch of size: {train_data[0].shape}')

    for i, nodule in enumerate(nodule_idxs):
        plt.subplot(len(train_label)//2, 2, 2*(i+1) - 1)
        plt.imshow(np.array(train_data[nodule][:, :, 1]), cmap='gray')
        plt.axis('off')

    for i, health in enumerate(health_idxs):
        plt.subplot(len(train_label) // 2, 2, 2*(i+1))
        plt.imshow(np.array(train_data[health][:, :, 1]), cmap='gray')
        plt.axis('off')

    g.show()
    plt.show()

if __name__ == "__main__":

    data_dir = '../../data/'

    patch_num = 10  # better divisible by 2
    train_data, labels = d.get_train_data_patient(data_dir, patient_num='LIDC-IDRI-0023',
                                                  patch_number=patch_num, tumor_rate=0.5)
    plot_data(train_data, labels)

    #test_patient = os.path.join(data_dir, 'raw', 'train', 'LIDC-IDRI-0666')
    #plot_patient(test_patient)
