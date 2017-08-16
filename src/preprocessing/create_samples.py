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
from tools.helper import convert_to_float, convert_to_floats

PATCH_SIZE_DEFAULT = [40, 40, 1]


def normalize(a):
    """
    Rescales the values of a given matrix to lie between 0 and 1.

    :param a: the matrix to be normalized
    :return: the normalized matrix
    """
    logger = logging.getLogger()
    a = a.astype(np.float32) # have to convert to float32 to avoid overflow
    a -= np.amin(a)
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
        process_data(data_dir, target='validation', patch_number=patch_number, patch_size=patch_size, tumor_rate=tumor_rate)

    else:
        if os.path.isdir(os.path.join(data_dir, 'processed', target)):
            shutil.rmtree(os.path.join(data_dir, 'processed', target))

        proc_data(data_dir, target, patch_number=patch_number, patch_size=patch_size, tumor_rate=tumor_rate)


def save_data(nodules, health, dir_path, patient_info_health, patient_info_nodule):
    """
    Puts the generated patches on disk separated for nodules and healthy patches

    :param nodules: the list of nodule patches
    :param health: the list of healthy patches
    :param dir_path: the target directory for the saved numpy arrays
    :param patient_info_health: a list with the patient number n times for each data point
    :param patient_info_nodule: a list with the patient number n times for each data point
    :return:
    """
    logger = logging.getLogger()
    logger.info('Saving patches for patient %s', patient_info_health[0])

    nod_path = os.path.join(dir_path, 'nodules')
    health_path = os.path.join(dir_path, 'health')

    os.makedirs(nod_path, exist_ok=True)
    os.makedirs(health_path, exist_ok=True)

    for i, nodule in enumerate(nodules):
        np.save(os.path.join(nod_path, patient_info_nodule[i]+'_'+str(i)), nodule)

    for i, health_patch in enumerate(health):
        np.save(os.path.join(health_path, patient_info_health[i]+'_'+str(i)), health_patch)


def proc_data(data_dir, target, *, patch_number=100, patch_size=PATCH_SIZE_DEFAULT, tumor_rate=0.5):
    """
    Reads all CT-Scans from a folder with several patients in it.

    :param data_dir: the path to the folder with the patient files
    :param target: the respective sub folder, either train test or validation
    :param patch_number: the number of patches per patient that should be retrieved
    :param patch_size: the 3d extend of the scan patches
    :param tumor_rate: the fraction of tumors that should be contained in the patches
    :return: data cubes and their labels
    """
    logger = logging.getLogger()

    path = os.path.join(data_dir, 'raw', target)
    path_result = os.path.join(data_dir, 'processed', target)

    logger.info('Start reading the data from %s', path)

    count = 0
    start_time = time.time()
    for dirName in os.listdir(path):
        folder = os.path.join(path, dirName)
        if os.path.isdir(folder):
            try:
                scans = read_patient(folder)
                annots = convert_to_floats(read_annotation(folder, scans))
                array_patient = conv2array(scans)

                data_nod, data_health = slice_patient(array_patient,
                                                      annots,
                                                      patch_size=patch_size,
                                                      number_of_patches=patch_number,
                                                      tumor_rate=tumor_rate)

                if np.amin(data_nod) < 0 or np.amin(data_health) < 0:
                    logger.error('Found negative data in patient %s, nodules: %f, health: %f', folder, np.amin(data_nod), np.amin(data_health))

                pat_info_health = [dirName] * len(data_nod)
                pat_info_nodule = [dirName] * len(data_health)
                count += 1
                save_data(data_nod, data_health, path_result, pat_info_health, pat_info_nodule)
            except AttributeError as e:
                # ignore secondary folders for one patient
                logger.error('Something went wrong with reading files from folder %s: %s', folder, e)
                continue

    logger.info('Read ct scan data from %s patients in %d minutes.', count, (time.time() - start_time)/60)


def read_patient(path):
    """
    Returns a sorted List of scans that are found in one specific folder.

    :param path: the path to the patient directory
    :return: all scans of this patient in the anatomic order
    """
    logger = logging.getLogger()
    all_scans = []
    for dirName, subdirList, fileList in os.walk(path):
        if len(fileList) > 20:  # there are smaller folders that can not be used since they lack the annotations
            for filename in fileList:
                if ".dcm" in filename.lower():
                    all_scans.append(dicom.read_file(os.path.join(dirName, filename)))

    if len(all_scans) == 0:
        logger.error('No scan files were read in folder %s', path)
        raise AttributeError('No scanfiles could be read')

    all_scans.sort(key=lambda x: int(x.InstanceNumber))
    return all_scans


def read_annotation(path, scan_files):
    """
    Reads the xml file associated with one patient.

    :param path: the path to the patient directory
    :param scan_files: the already read in slice data
    :return: a list of all nodule locations in x y z locations
    """
    logger = logging.getLogger()
    logger.info('Read annotations from: %s', path)
    nodule_locations = []
    ref_scan_1 = scan_files[0]
    ref_scan_2 = scan_files[1]

    slice_begin = convert_to_float(ref_scan_1.ImagePositionPatient[2])
    ref2 = ref_scan_2.ImagePositionPatient[2]
    ref1 = ref_scan_1.ImagePositionPatient[2]
    slice_distance = np.abs(convert_to_float(ref2) - convert_to_float(ref1))

    for dirName, subdirList, fileList in os.walk(path):
        if len(fileList) > 20: # in the case of two folders take the larger one
            for filename in fileList:
                if '.xml' in filename.lower():
                    xml_anno = xml.etree.ElementTree.parse(os.path.join(dirName, filename)).getroot()
                    break  # there is only one xml file per patient
    domain = '{http://www.nih.gov}'
    for session in xml_anno.findall(domain + 'readingSession'):
        for nodule in session.findall(domain + 'unblindedReadNodule'):

            mapCount = 0
            roiCount = 0
            xPosAll = 0
            yPosAll = 0
            zPosAll = 0

            for roi in nodule.findall(domain + 'roi'):
                roiCount = roiCount + 1
                zPos_tmp = convert_to_float(roi.find(domain + 'imageZposition').text)
                ztransformed = np.abs(convert_to_float(zPos_tmp) - slice_begin) // slice_distance
                zPosAll = zPosAll + ztransformed

                for edgeMap in roi.findall(domain + 'edgeMap'):
                    mapCount = mapCount + 1
                    xPosAll = xPosAll + convert_to_float(edgeMap.find(domain + 'xCoord').text)
                    yPosAll = yPosAll + convert_to_float(edgeMap.find(domain + 'yCoord').text)

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


def slice_patient(all_scans, annotation, patch_size=PATCH_SIZE_DEFAULT, number_of_patches=100, tumor_rate=0.3):
    """
    Generates cubes from a complete CT-Scan with a fixed size and a given distribution of
    tumor containing patches.

    :param all_scans: a complete CT-Scan of a patient
    :param annotation: the annotation data for this patient as a list of floats
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
        tumor = [int(x) for x in random.choice(annotation)]
        i = 0
        while((int(tumor[0] - patch_size[0]) < 0
              or int(tumor[0] + patch_size[0]) > all_scans.shape[0]
              or int(tumor[1] - patch_size[1]) < 0
              or int(tumor[1] + patch_size[1]) > all_scans.shape[1]
              or int(tumor[2] - patch_size[2]) < 0
              or int(tumor[2] + patch_size[2]) > all_scans.shape[2])
              and i < 100
              ):
            i = i + 1
            tumor = [int(x) for x in random.choice(annotation)]  # take another annotation and hope for the best

        if i > 99:
            logger.error(f'Could not find a matching patch in {i} tries, aborting ...')
            raise AttributeError('Nodules do not fit!')

        # use random start point around nodule
        # x = random.randint(tumor[0] - patch_size[0], tumor[0])
        # y = random.randint(tumor[1] - patch_size[1], tumor[1])
        # z = random.randint(tumor[2] - patch_size[2], tumor[2])

        # center nodule
        x = tumor[0] - patch_size[0]//2
        y = tumor[1] - patch_size[1]//2
        z = tumor[2] - patch_size[2]//2
        start_point = [x, y, z]

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

        # check patch size
        if np.array(patch).shape != tuple(patch_size):
            continue  # sometimes it is just not right and python doesn't complain

        for anno in annotation:
            if anno in patch:
                continue  # try again to find a patch

        healthy_patches.append(patch)

    return np.array(nodule_patches), np.array(healthy_patches)
