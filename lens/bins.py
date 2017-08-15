import numpy as np


def freedman_diaconis_bins(data_range, iqr, number_samples):
    """
    Calculate number of hist bins using Freedman-Diaconis rule.

    Parameters
    ----------

    data_range: float
        total range of the data

    iqr: float
        interquartile range of the data

    number_samples: int
        number of values in the data
    """
    # From http://stats.stackexchange.com/questions/798/
    # adapted from seaborn
    h = 2 * iqr / (float(number_samples) ** (1. / 3.))
    # fall back to sqrt(a) bins if iqr is 0
    if h == 0:
        return int(np.sqrt(number_samples))
    else:
        return int(np.ceil(data_range) / h)
