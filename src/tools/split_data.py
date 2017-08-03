import os
import errno
import numpy as np
import shutil

from random import shuffle


def create_folder(path):
    """
    Creates a folder if it doesn't exist.
    :param path: the path to be created.
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def split_folder(path, *, ratio=[.8, .1, .1]):
    """
    Creates appropriate sub folders and uses those to split the dataset into
    train/validation/test. The files are moved to the appropriate folders.
    
    :param path: the base path that contains the files.
    :param ratio: the ratio to be split to
    """
    all_data = os.listdir(path)
    all_data = [file for file in all_data if file.startswith('LIDC')]
    shuffle(all_data)
    train, validation, test = np.split(np.array(all_data),
                                     [int(ratio[0] * len(all_data)), int((ratio[0]+ratio[1]) * len(all_data))])
    print(f'Length of train, validate, test : {len(train)}, {len(validation)}, {len(test)}')

    # create subfolders
    train_fold = os.path.join(path, 'train')
    test_fold = os.path.join(path, 'test')
    val_fold = os.path.join(path, 'validation')

    create_folder(train_fold)
    create_folder(test_fold)
    create_folder(val_fold)

    print('Start copying files ...')
    for file in train:
        shutil.move(os.path.join(path, file), os.path.join(train_fold, file))

    for file in test:
        shutil.move(os.path.join(path, file), os.path.join(test_fold, file))

    for file in validation:
        shutil.move(os.path.join(path, file), os.path.join(val_fold, file))


if __name__ == "__main__":

    folder = input('Please specify the target directory: \n')
    train = float(input('How many percent should be training data? \n'))
    if any(folder in exit_term for exit_term in {'q', 'e', 'quit', 'Q'}):
         print('Terminating...')
         exit()
    rest = (1.0-train)/2
    split_folder(folder, ratio=[train, rest, rest])
    print('Done!')
