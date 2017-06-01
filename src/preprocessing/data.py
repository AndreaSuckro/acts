import xml.etree.ElementTree
import numpy as np
import dicom
import os
from cvloop import cvloop
import time
import random
import logging

def get_train_data(data_dir, *, patch_number=100, patch_size=[50, 50, 3], tumor_rate=0.3):
    """
    Reads all training data as specified in the training data.

    :param data_dir: the directory that contains a folder for the train data
    :param patch_number: the number of patches per patient that should be retrieved
    :param patch_size: the 3d extend of the scan patches
    :param tumor_rate: the fraction of tumors that should be contained in the patches
    :return: data with nodules and without
    """
    return get_data(os.path.join(data_dir, 'train'),
                    patch_number=patch_number,
                    patch_size=patch_size,
                    tumor_rate=tumor_rate)


def get_test_data(data_dir, *, patch_number=100, patch_size=[50, 50, 3], tumor_rate=0.3):
    """
    Returns all data under the test directory.

    :param data_dir: the directory that contains a folder for the test data
    :param patch_number: the number of patches per patient that should be retrieved
    :param patch_size: the 3d extend of the scan patches
    :param tumor_rate: the fraction of tumors that should be contained in the patches
    :return: data with nodules and without
    """
    return get_data(os.path.join(data_dir, 'test'),
                    patch_number=patch_number,
                    patch_size=patch_size,
                    tumor_rate=tumor_rate)


def get_data(path, *, patch_number=100, patch_size=[50, 50, 3], tumor_rate=0.3):
    """
    Reads all CT-Scans from a folder with several patients in it.

    :param path: the path to the folder with the patient files
    :param patch_number: the number of patches per patient that should be retrieved
    :param patch_size: the 3d extend of the scan patches
    :param tumor_rate: the fraction of tumors that should be contained in the patches
    :return: datacubes and their labels
    """
    logger = logging.getLogger()

    data_nod_all = []
    data_health_all = []
    count = 0
    start_time = time.time()
    for dirName in os.listdir(path):
        folder = os.path.join(path, dirName)
        if os.path.isdir(folder):
            scans = read_patient(folder)
            try:
                annots = read_annotation(folder, scans)
                array_patient = conv2array(scans)
                data_nod, data_health = slice_patient(array_patient,
                                                      annots,
                                                      patch_size=patch_size,
                                                      number_of_patches=patch_number,
                                                      tumor_rate=tumor_rate)
                data_nod_all += data_nod
                data_health_all += data_health
                count += 1
            except AttributeError as e:
                # ignore secondary folders for one patient
                logger.error('Something went wrong with reading files from folder %s: %s', folder, e)
                continue

    logger.info('Read ct scan data from %s patients in %d seconds.', count, time.time() - start_time)
    data = data_nod_all + data_health_all
    labels_nod = [1] * len(data_nod_all)
    labels_health = [0] * len(data_health_all)
    labels = labels_nod + labels_health

    combined = list(zip(data, labels))
    random.shuffle(combined)
    logger.info('Successfully read in %s lung patches.', len(data))

    return zip(*combined)


def read_patient(path):
    """
    Returns a sorted List of scans that are found in one specific folder.

    :param path: the path to the patient directory
    :return: all scans of this patient in the anatomic order
    """
    all_scans = []
    for dirName, subdirList, fileList in os.walk(path):
        if len(fileList) > 20:  # there are smaller folders that can not be used since they lack the annotations
            for filename in fileList:
                if ".dcm" in filename.lower():
                    all_scans.append(dicom.read_file(os.path.join(dirName, filename)))
    all_scans.sort(key=lambda x: int(x.InstanceNumber))
    return all_scans


def read_annotation(path, scan_files):
    """
    Reads the xml file associated with one patient.

    :param path: the path to the patient directory
    :param scan_files: the already read in slice data
    :return: a list of all nodule locations in x y z locations
    """

    nodule_locations = []

    ref_scan_1 = scan_files[0]
    ref_scan_2 = scan_files[1]

    slice_begin = float(ref_scan_1.ImagePositionPatient[2])
    slice_distance = np.abs(float(ref_scan_2.ImagePositionPatient[2]) - float(ref_scan_1.ImagePositionPatient[2]))

    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            if '.xml' in filename.lower():
                xml_anno = xml.etree.ElementTree.parse(os.path.join(dirName, filename)).getroot()
                break  # there is only one xml file per patient

    for session in xml_anno.findall('{http://www.nih.gov}readingSession'):
        for nodule in session.findall('{http://www.nih.gov}unblindedReadNodule'):
            xPos = float(nodule.find('{http://www.nih.gov}roi').find('{http://www.nih.gov}edgeMap')
                         .find('{http://www.nih.gov}xCoord').text)
            yPos = float(nodule.find('{http://www.nih.gov}roi').find('{http://www.nih.gov}edgeMap')
                         .find('{http://www.nih.gov}yCoord').text)
            zPos = float(nodule.find('{http://www.nih.gov}roi').find('{http://www.nih.gov}imageZposition').text)

            ztransformed = np.abs(float(zPos) - slice_begin) // slice_distance

            nodule_locations.append(np.array([xPos, yPos, ztransformed]))  # beware of switched x,y!

    return nodule_locations


def normalize(a):
    """
    Rescales the values of a given matrix to lie between 0 and 1.

    :param a: the matrix to be normalized
    :return: the normalized matrix
    """
    a += np.abs(np.amin(a))
    a = a / np.amax(a)
    return a


def conv2array(scan_files):
    """
    Converts the dicom files of one patient to a numpy array.
    """
    ref_scan_first = scan_files[0]

    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    pixel_dims = (int(ref_scan_first.Rows), int(ref_scan_first.Columns), len(scan_files))

    # The array is sized based on 'pixel_dims'
    array_imgs = np.zeros(pixel_dims, dtype=ref_scan_first.pixel_array.dtype)

    # loop through all the DICOM files
    for i in range(len(scan_files)):
        # store the raw image data
        array_imgs[:, :, i] = scan_files[i].pixel_array
    return normalize(array_imgs)


def slice_patient(all_scans, annotation, patch_size=[50, 50, 3], number_of_patches=100, tumor_rate=0.3):
    """
    Generates cubes from a complete CT-Scan with a fixed size and a given distribution of
    tumor containing patches.

    :param all_scans: a complete CT-Scan of a patient
    :param annotation: the annotation data for this patient
    :param patch_size: the size of the cubes in [x,y,z] format
    :param number_of_patches: the number of cubes to be cut
    :param tumor_rate: the amount of tumorous patches
    :return: the generated slices in a list with their classification
    """

    # get all the tumor patches
    nodule_patches = []
    for i in range(int(number_of_patches*tumor_rate)):
        # pick a random tumor
        tumor = random.choice(annotation)
        start_point = [random.randint(tumor[0] - patch_size[0], tumor[0]),
                       random.randint(tumor[1] - patch_size[1], tumor[1]),
                       random.randint(tumor[2] - patch_size[2], tumor[2])]

        nodule_patches.append(all_scans[start_point[0]:start_point[0] + patch_size[0],
                                        start_point[1]:start_point[1] + patch_size[1],
                                        start_point[2]:start_point[2] + patch_size[2]])

    # get all the regular patches
    healthy_patches = []
    while len(healthy_patches) < (number_of_patches - int(number_of_patches*tumor_rate)):
        # check that coordinates do not overlap tumor?
        start_point = [random.randint(0, all_scans.shape[0] - patch_size[0]),
                       random.randint(0, all_scans.shape[1] - patch_size[1]),
                       random.randint(0, all_scans.shape[2] - patch_size[2])]

        patch = all_scans[start_point[0]:start_point[0] + patch_size[0],
                          start_point[1]:start_point[1] + patch_size[1],
                          start_point[2]:start_point[2] + patch_size[2]]

        for anno in annotation:
            if anno in patch:
                continue  # try again to find a patch

        healthy_patches.append(patch)

    return [nodule_patches, healthy_patches]


if __name__ == "__main__":
    # show case the methods of the data module
    testPatient = "../../data/train/LIDC-IDRI-0666/"
    scan_pat = read_patient(testPatient)
    annos = read_annotation(testPatient, scan_pat)
    array_patient = conv2array(scan_pat)
    print(f'Shape of scan data: {array_patient.shape}')

    data_nod, data_health = slice_patient(array_patient, annos)
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

    cvloop(Data(array_patient), annotations=format_list, print_info=True)
