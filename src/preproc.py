from preprocessing.create_samples import process_data
from preprocessing.create_samples import PATCH_SIZE_DEFAULT

import logging.config
import os
import argparse

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
    logging.config.fileConfig(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + 'logging.ini')

    logger = logging.getLogger()
    main(**parse_args())
