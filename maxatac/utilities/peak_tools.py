import pandas as pd
import numpy as np


def get_genomic_locations(array, threshold, chromosome, bin_size):
    """
    Get the genomic locations that correspond to bins greater than or equal to some threshold
    """

    # Find threshold around the given recall or precision
    target_bin_idx_list = np.argwhere(array >= threshold)

    BIN_list = []

    for prediction_bin in target_bin_idx_list:
        start = prediction_bin[0]*bin_size
        BIN_list.append([chromosome, start, start+bin_size])

    return pd.DataFrame(BIN_list, columns=["chr", "start", "end"])
