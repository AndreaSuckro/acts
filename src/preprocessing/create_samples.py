import xml.etree.ElementTree
import numpy as np
import dicom
import os
import time
import random
import logging.config
import shutil
import argparse
import sys

PATCH_SIZE_DEFAULT = [50, 50, 3]


def normalize(a):
    """
    Rescales the values of a given matrix to lie between 0 and 1.

    :param a: the matrix to be normalized
    :return: the normalized matrix
    """
    a += np.abs(np.amin(a))
    a = a / np.amax(a)
    return a


def process_data(data_dir, *, target='all', patch_number=100, patch_size=PATCH_SIZE_DEFAULT, tumor_rate=0.5):
    """
    Reads all training data as specified in the training data.

    :param data_dir: the directory that contains the raw folder for the patient data
    :param target: used to specify whether training, test or validation dataset should be used
    :param patch_number: the number of patches PER PATIENT that should be retrieved
    :param patch_size: the 3d extend of the scan patches
    :param tumor_rate: the fraction of tumors that should be contained in the patches
    """

    if target == 'all':  # if all data should be preprocessed just call the method again
        process_data(data_dir, target='train', patch_number=patch_number, patch_size=patch_size, tumor_rate=tumor_rate)
        process_data(data_dir, target='test', patch_number=patch_number, patch_size=patch_size, tumor_rate=tumor_rate)
        process_data(data_dir, target='validate', patch_number=patch_number, patch_size=patch_size, tumor_rate=tumor_rate)

    else:
        if os.path.isdir(os.path.join(data_dir, 'processed', target)):
            shutil.rmtree(os.path.join(data_dir, 'processed', target))

        nodules, health, patient_info_health, patient_info_nodule = get_data(os.path.join(data_dir, 'raw', target),
                                                                             patch_number=patch_number,
                                                                             patch_size=patch_size,
                                                                             tumor_rate=tumor_rate)

        save_data(nodules, health, os.path.join(data_dir, 'processed', target), patient_info_health, patient_info_nodule)


def save_data(nodules, health, dir_path, patient_info_health, patient_info_nodule):
    """
    Puts the generated patches on disk separated for nodules and healthy patches
    
    :param nodules: the list of nodule patches
    :param health: the list of healthy patches
    :param dir_path: the target directory for the saved numpy arrays 
    :param patient_info: number of the patient for each sample
    :return: 
    """

    nod_path = os.path.join(dir_path, 'nodules')
    health_path = os.path.join(dir_path, 'health')

    os.makedirs(nod_path, exist_ok=True)
    os.makedirs(health_path, exist_ok=True)

    for i, nodule in enumerate(nodules):
        np.save(os.path.join(nod_path, patient_info_nodule[i]+'_'+str(i)), nodule)

    for i, health_patch in enumerate(health):
        np.save(os.path.join(health_path, patient_info_health[i]+'_'+str(i)), health_patch)


def get_data(path, *, patch_number=100, patch_size=PATCH_SIZE_DEFAULT, tumor_rate=0.5):
    """
    Reads all CT-Scans from a folder with several patients in it.

    :param path: the path to the folder with the patient files
    :param patch_number: the number of patches per patient that should be retrieved
    :param patch_size: the 3d extend of the scan patches
    :param tumor_rate: the fraction of tumors that should be contained in the patches
    :return: datacubes and their labels
    """
    logger = logging.getLogger()

    logger.info('Start reading the data from %s', path)

    data_nod_all = []
    data_health_all = []
    pat_info_health = []
    pat_info_nodule = []
    count = 0
    start_time = time.time()
    for dirName in os.listdir(path):
        folder = os.path.join(path, dirName)
        if os.path.isdir(folder):
            scans = read_patient(folder)
            try:
                annots = read_annotation(folder, scans)
                array_patient = conv2array(scans)

                if annots is None or len(annots) == 0:
                    raise AttributeError('Annotations contain no Nodules')

                data_nod, data_health = slice_patient(array_patient,
                                                      annots,
                                                      patch_size=patch_size,
                                                      number_of_patches=patch_number,
                                                      tumor_rate=tumor_rate)
                data_nod_all += data_nod
                data_health_all += data_health
                pat_info_health += [dirName] * len(data_nod)
                pat_info_nodule += [dirName] * len(data_health)
                count += 1
            except AttributeError as e:
                # ignore secondary folders for one patient
                logger.error('Something went wrong with reading files from folder %s: %s', folder, e)
                continue

    logger.info('Read ct scan data from %s patients in %d seconds.', count, time.time() - start_time)

    return data_nod_all, data_health_all, pat_info_health, pat_info_nodule


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

            mapCount = 0
            roiCount = 0
            xPosAll = 0
            yPosAll = 0
            zPosAll = 0

            for roi in nodule.findall('{http://www.nih.gov}roi'):
                roiCount = roiCount + 1
                zPos_tmp = float(roi.find('{http://www.nih.gov}imageZposition').text)
                ztransformed = np.abs(float(zPos_tmp) - slice_begin) // slice_distance
                zPosAll = zPosAll + ztransformed

                for edgeMap in roi.findall('{http://www.nih.gov}edgeMap'):
                    mapCount = mapCount + 1
                    xPosAll = xPosAll + float(edgeMap.find('{http://www.nih.gov}xCoord').text)
                    yPosAll = yPosAll + float(edgeMap.find('{http://www.nih.gov}yCoord').text)

            xPos = xPosAll/mapCount
            yPos = yPosAll/mapCount
            zPos = zPosAll/roiCount

            nodule_locations.append(np.array([xPos, yPos, zPos]))  # beware of switched x,y!

    return nodule_locations


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
    logger = logging.getLogger()
    nodule_patches = []

    for i in range(int(number_of_patches*tumor_rate)):
        # pick a random tumor
        tumor = random.choice(annotation)

        # use random start point around nodule
        #start_point = [random.randint(tumor[0] - patch_size[0], tumor[0]),
        #               random.randint(tumor[1] - patch_size[1], tumor[1]),
        #               random.randint(tumor[2] - patch_size[2], tumor[2])]

        # center nodule
        start_point = [int(tumor[0] - patch_size[0]//2),
                       int(tumor[1] - patch_size[1]//2),
                       int(tumor[2] - patch_size[2]//2)]

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


def main(data_dir=None, target='all', patch_number=100, patch_size=PATCH_SIZE_DEFAULT, tumor_rate=0.5, show_case=False):
    """
    Processes the data and shows a sample if needed. See the documentation of process_data() for more information
    on the parameters.
    """
    process_data(data_dir, target=target, patch_number=patch_number, patch_size=patch_size, tumor_rate=tumor_rate)


def parse_args():
    """
    Parses the arguments for the main call of this function.
    :return: a dictionary of arguments
    """
    parser = argparse.ArgumentParser(description='Data processing for the lung CTs of patients.')
    parser.add_argument('-d', '--data', dest='data_dir', help='Path to the data directory, this is required')
    parser.add_argument('-p', '--patch_num', dest='patch_number',
                        default=100, type=int, help='Number of patches to be loaded per patient')
    parser.add_argument('-t', '--target', dest='target',
                        default='all', help='Choose whether all, train, test or validation should be used')
    parser.add_argument('-s', '--patch_size', dest='patch_size',
                        default=PATCH_SIZE_DEFAULT, type=list, help='Size of the patches to be generated in [x,y,z]')
    parser.add_argument('-r', '--tumor_rate', dest='tumor_rate',
                        default=0.5, type=float, help='The fraction of the tumorous patches to be generated')
    parser.add_argument('-f', '--plot_figure', dest='show_case',
                        default=False, type=bool, help='True when a sample should be plotted')
    args = parser.parse_args()

    if not args.data_dir:
        sys.exit(parser.print_help())

    return vars(args)


if __name__ == '__main__':
    """
    Just calls the main method with the parameters defined over the commandline interface.
    """
    # initialize logging
    logging.config.fileConfig(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'logging.ini'))

    logger = logging.getLogger()
    main(**parse_args())
