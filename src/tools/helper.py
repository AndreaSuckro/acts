import numpy as np


def convert_to_float(val):
    """
    Tries to convert one value to float if possible.
    """
    if is_convertable(val, float):
        return float(val)
    else:
        raise AttributeError('Converted value is empty for this patient')


def convert_to_floats(annots):
    """
    Tries to convert a list of lists of whatever to a list of list of floats.
    """
    if annots is None or len(annots) == 0:
        raise AttributeError('Annotations are empty for this patient')

    # try converting to float
    annof = [[float(y) for y in x if is_convertable(y, float)] for x in annots]
    if annof is None or len(annof) == 0:
        raise AttributeError('Converted valued are empty for this patient')
    return annof


def is_convertable(value, type):
    """
    Checks whether conversion to this type works, raising NAN values as error.
    """
    try:
        res = type(value)
        if np.isnan(res):
            return False
        return True
    except (ValueError, TypeError) as error:
        print(value)
        return False
