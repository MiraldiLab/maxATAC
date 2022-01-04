import numpy as np
import pyBigWig
from sklearn import metrics


def import_blacklist_mask(bigwig_path, chromosome, chromosome_length, bin_count):
    """
        Import the chromosome signal from a blacklist bigwig file and convert to a numpy array to use to mask out
        the regions to exclude in the AUPR analysis
        :return: blacklist_mask: A np.array the has True for regions that should be excluded from analysis
        """
    with pyBigWig.open(bigwig_path) as input_bw:
        return np.array(input_bw.stats(chromosome,
                                       0,
                                       chromosome_length,
                                       type="max",
                                       nBins=bin_count
                                       ),
                        dtype=float  # need it to have NaN instead of None
                        ) != 1  # Convert to boolean array, select areas that are not 1


def import_GoldStandard_array(bigwig_path, chromosome, chromosome_length, bin_count):
    with pyBigWig.open(bigwig_path) as input_bw:
        return np.nan_to_num(np.array(input_bw.stats(chromosome,
                                                     0,
                                                     chromosome_length,
                                                     type="max",
                                                     nBins=bin_count,
                                                     exact=True
                                                     ),
                                      dtype=float  # need it to have NaN instead of None
                                      )
                             ) > 0  # to convert to boolean array


def calculate_AUC_per_rank(PR_CURVE_DF, threshold):
        """
        Calculate the AUC at each rank on the AUPRC curve
        """
        tmp_df = PR_CURVE_DF[PR_CURVE_DF["Threshold"] >= threshold]

        # If we only have 1 point do not calculate AUC
        if len(tmp_df["Threshold"].unique()) == 1:
            return 0
        else:
            return metrics.auc(y=tmp_df["Precision"], x=tmp_df["Recall"])