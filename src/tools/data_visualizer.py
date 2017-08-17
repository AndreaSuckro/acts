import matplotlib.pyplot as plt
from matplotlib import image
import preprocessing.create_samples as data
from preprocessing.data import get_train_data, get_validation_data
from cvloop import cvloop
import time
import numpy as np


class DataVisualizer:
    """
    Is used to visualize the data, the network is trained with.
    Shows healthy patches on the left and patches with nodules on the right side.
    """
    def __init__(self, data, labels, *,
                 rows=1,
                 cmaps=['gray'],
                 name='Healthy vs. Nodule Patches'):

        self.data = data
        self.rows = rows
        self.current = -1

        self.nodule_idxs = [i for i, x in enumerate(labels) if x == 1]
        self.health_idxs = [i for i, x in enumerate(labels) if x == 0]

        self.pos_health = 0 # for the 3D component
        self.pos_nodule = 0

        self.figure, self.axes = plt.subplots(rows, 2)
        if name:
            self.figure.canvas.set_window_title(name)

        self.axes = self.axes.flatten()
        self.images = []
        for i, axes in enumerate(self.axes):
            img = image.AxesImage(axes, cmap=cmaps[0])
            axes.set_aspect('equal')
            example = data[0]
            axes.set_xlim([0, example.shape[1]])
            axes.set_ylim([example.shape[0], 0])
            self.images.append(axes.add_image(img))

        self.show_next()

        self.keycb = self.figure.canvas.mpl_connect(
                'key_press_event',
                lambda event: self.__key_press_event(event))

    def show_next(self):
        self.current = self.current + 1
        self.update_axes()

    def show_up_nodule(self):
        self.pos_nodule = (self.pos_nodule + 1)%self.data[0].shape[2] # shape is the same for health and nodule
        self.update_axes()

    def show_up_health(self):
        self.pos_health = (self.pos_health + 1)%self.data[0].shape[2] # shape is the same for health and nodule
        self.update_axes()

    def show_previous(self):
        self.current = (self.current - 1) % len(self.nodule_idxs)
        self.update_axes()

    def update_axes(self):
        first = self.current % len(self.nodule_idxs)

        nodule_idx = self.nodule_idxs[self.current % len(self.nodule_idxs)]
        health_idx = self.health_idxs[self.current % len(self.health_idxs)]

        self.images[0].set_data(self.data[health_idx][...,self.pos_health])
        self.images[1].set_data(self.data[nodule_idx][...,self.pos_nodule])

        self.figure.suptitle(f'Showing health vs. nodule samples {first} to {first + 1} from {len(self.data)}')
        self.figure.canvas.draw()

    def __key_press_event(self, event):
        events = {
            'q': lambda event: plt.close(self.figure),
            'escape': lambda event: plt.close(self.figure),
            'cmd+w': lambda event: plt.close(self.figure),
            'right': lambda event: self.show_next(),
            'down': lambda event: self.show_next(),
            'left': lambda event: self.show_previous(),
            'up': lambda event: self.show_previous(),
            'h': lambda event: self.show_up_health(),
            'n': lambda event: self.show_up_nodule()
        }
        try:
            events[event.key](event)
        except KeyError:
            print(f'Key pressed but no action available: {event.key}')


def plot_patient(path):
    """
    Plots all scan data for one patient in a loop.

    :param path: Path to the data directory.
    """
    # show case the methods of the data module
    scan_pat = data.read_patient(path)
    annos = data.read_annotation(path, scan_pat)
    array_patient = data.conv2array(scan_pat)
    print(f'Shape of scan data: {array_patient.shape}')

    data_nod, data_health = data.slice_patient(array_patient, annos)
    print(f'Shape of nodules: {len(data_nod)}x{data_nod[0].shape}, '
          f'Shape of health: {len(data_health)}x{data_health[0].shape}')
    print(f'Annotations for this patient: {annos}')
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
            time.sleep(0.5)
            self.i = (self.i + 1) if self.i < self.scans.shape[2] - 1 else 0
            img = self.scans[:, :, self.i]
            return True, img

    cvloop(Data(array_patient), annotations=annos, print_info=True)


def plot_histogram(data_dir):
    """
    Plots the data distribution of validation and training.

    :param data_dir: the data directory
    :return: a plot for the data distribution
    """
    train_data_raw, train_labels_raw = get_train_data(data_dir, patch_number=1000)
    train_data_raw = np.array(train_data_raw)
    validation_data_raw, validation_labels_raw = get_validation_data(data_dir, patch_number=1000)
    validation_data_raw = np.array(validation_data_raw)

    train_nodule_idxs = [i for i, x in enumerate(train_labels_raw) if x == 1]
    train_health_idxs = [i for i, x in enumerate(train_labels_raw) if x == 0]

    validation_nodule_idxs = [i for i, x in enumerate(validation_labels_raw) if x == 1]
    validation_health_idxs = [i for i, x in enumerate(validation_labels_raw) if x == 0]

    train_nodules = np.array(train_data_raw[train_nodule_idxs].flatten())
    train_health = np.array(train_data_raw[train_health_idxs].flatten())
    validation_nodules = np.array(validation_data_raw[validation_nodule_idxs].flatten())
    validation_health = np.array(validation_data_raw[validation_health_idxs].flatten())

    f, axarr = plt.subplots(2, 2)
    axarr = axarr.ravel()
    axarr[0].hist(train_health, bins=30)
    axarr[0].set_title("Train")
    axarr[0].set_ylabel("Health")
    axarr[0].get_yaxis().set_ticks([])

    axarr[2].hist(train_nodules, bins=30)
    axarr[2].set_ylabel("Nodule")
    axarr[2].get_yaxis().set_ticks([])

    axarr[1].hist(validation_health, bins=30)
    axarr[1].set_title("Validation")
    axarr[1].get_yaxis().set_ticks([])

    axarr[3].hist(validation_nodules, bins=30)
    axarr[3].get_yaxis().set_ticks([])


if __name__ == "__main__":

    data_dir = '../../data/'

    plot_histogram(data_dir)
