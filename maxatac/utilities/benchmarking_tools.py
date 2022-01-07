import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from maxatac.utilities.genome_tools import load_bigwig, chromosome_blacklist_mask
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from scipy import stats
from maxatac.utilities.system_tools import remove_tags
import pybedtools
from maxatac.utilities.constants import DEFAULT_CHROM_SIZES as chrom_sizes
from maxatac.utilities.constants import BLACKLISTED_REGIONS  as blacklist_bed_location


class ChromosomeAUPRC(object):
    """
    Benchmark maxATAC binary predictions against a gold standard using AUPRC.

    During initialization the following steps will be performed:

    1) Set up run parameters and calculate bins needed
    2) Load bigwig files into np.arrays
    3) Calculate AUPRC stats
    """

    def __init__(self,
                 prediction_bw,
                 goldstandard_bw,
                 blacklist_bw,
                 chromosome,
                 bin_size,
                 agg_function,
                 results_location,
                 round_predictions
                 ):
        """
        :param prediction_bw: Path to bigwig file containing maxATAC predictions
        :param goldstandard_bw: Path to gold standard bigwig file
        :param blacklist_bw: Path to blacklist bigwig file
        :param chromosome: Chromosome to benchmark
        :param bin_size: Resolution to bin the results to
        :param agg_function: Method to use to aggregate multiple signals in the same bin
        """
        self.results_location = results_location

        self.prediction_stream = load_bigwig(prediction_bw)
        self.goldstandard_stream = load_bigwig(goldstandard_bw)

        self.chromosome = chromosome
        self.chromosome_length = self.goldstandard_stream.chroms(self.chromosome)

        self.bin_count = int(int(self.chromosome_length) / int(bin_size))  # need to floor the number
        self.bin_size = bin_size

        self.agg_function = agg_function

        self.blacklist_mask = chromosome_blacklist_mask(blacklist_bw,
                                                        self.chromosome,
                                                        self.chromosome_length,
                                                        self.bin_count)

        self.__import_prediction_array__(round_prediction=round_predictions)
        self.__import_goldstandard_array__()
        self.__AUPRC__()
        self.__plot()

    def __import_prediction_array__(self, round_prediction=6):
        """
        Import the chromosome signal from the predictions bigwig file and convert to a numpy array.

        :param round_prediction: The number of floating places to round the signal to
        :return: prediction_array: A np.array that has values binned according to bin_count and aggregated according
        to agg_function
        """
        logging.error("Import Predictions Array")

        # Get the bin stats from the prediction array
        self.prediction_array = np.nan_to_num(np.array(self.prediction_stream.stats(self.chromosome,
                                                                                    0,
                                                                                    self.chromosome_length,
                                                                                    type=self.agg_function,
                                                                                    nBins=self.bin_count,
                                                                                    exact=True),
                                                       dtype=float  # need it to have NaN instead of None
                                                       )
                                              )

        self.prediction_array = np.round(self.prediction_array, round_prediction)

    def __import_goldstandard_array__(self):
        """
        Import the chromosome signal from the gold standard bigwig file and convert to a numpy array with True/False
        entries.

        :return: goldstandard_array: A np.array has values binned according to bin_count and aggregated according to
        agg_function. random_precision: The random precision of the model based on # of True bins/ # of genomic bins
        """
        logging.error("Import Gold Standard Array")

        # Get the bin stats from the gold standard array
        self.goldstandard_array = np.nan_to_num(np.array(self.goldstandard_stream.stats(self.chromosome,
                                                                                        0,
                                                                                        self.chromosome_length,
                                                                                        type=self.agg_function,
                                                                                        nBins=self.bin_count,
                                                                                        exact=True
                                                                                        ),
                                                         dtype=float  # need it to have NaN instead of None
                                                         )
                                                ) > 0  # to convert to boolean array

        self.random_precision = np.count_nonzero(self.goldstandard_array[self.blacklist_mask]) / \
                                np.size(self.prediction_array[self.blacklist_mask])

    def __get_true_positives__(self, threshold):
        """
        Get the number of true positives predicted at a given threshold

        :param threshold: The desired value threshold to limit analysis to
        :return: Number of true positives predicted by the model
        """
        # Find the idxs for the bins that are gt/et some threshold
        tmp_prediction_idx = np.argwhere(self.prediction_array >= threshold)

        # Find the bins in the gold standard that match the threshold prediction bins
        tmp_goldstandard_threshold_array = self.goldstandard_array[tmp_prediction_idx]

        # Count the number of bins in the intersection that are True
        return len(np.argwhere(tmp_goldstandard_threshold_array == True))

    def __get_false_positives__(self, threshold):
        """
        Get the number of false positives predicted at a given threshold

        :param threshold: The desired value threshold to limit analysis to
        :return: Number of false positives predicted by the model
        """
        # Find the idxs for the bins that are gt/et some threshold
        tmp_prediction_idx = np.argwhere(self.prediction_array >= threshold)

        # Find the bins in the gold standard that match the thresholded prediction bins
        tmp_goldstandard_threshold_array = self.goldstandard_array[tmp_prediction_idx]

        # Count the number of bins in the intersection that are False
        return len(np.argwhere(tmp_goldstandard_threshold_array == False))

    def __get_bin_count__(self, threshold):
        """
        Get the number of bins from the prediction array that are greater than or equal to some threshold
        """
        return len(self.prediction_array[self.prediction_array >= threshold])

    def __calculate_AUC_per_rank__(self, threshold):
        """
        Calculate the AUC at each rank on the AUPRC curve
        """
        tmp_df = self.PR_CURVE_DF[self.PR_CURVE_DF["Threshold"] >= threshold]

        # If we only have 1 point do not calculate AUC
        if len(tmp_df["Threshold"].unique()) == 1:
            return 0
        else:
            return metrics.auc(y=tmp_df["Precision"], x=tmp_df["Recall"])

    def __AUPRC__(self):
        """
        Calculate the AUPRc for the predictions compared to a gold standard

        This function will perform the following steps:

        1) AUPR analysis. The sklearn documents states that there are 1 extra set of points added to the curve. We
        remove the last point added to the curve.
        2) Calculate the AUC for each threshold for visualization
        3) Generate statistics for each threshold: tp, fp, fn
        4) Write tsv of AUPR file stats

        :return: AUPRC stats as a pandas dataframe
        """
        logging.error("Calculate precision-recall curve for " + self.chromosome)

        self.precision, self.recall, self.thresholds = precision_recall_curve(
            self.goldstandard_array[self.blacklist_mask],
            self.prediction_array[self.blacklist_mask])

        logging.error("Making DataFrame from results")

        # Create a dataframe from the results
        # Issue 54:
        # The sklearn package will add a point at precision=1 and recall=0
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
        # remove the last point of the array which corresponds to this extra point
        self.PR_CURVE_DF = pd.DataFrame(
            {'Precision': self.precision[:-1], 'Recall': self.recall[:-1], "Threshold": self.thresholds})

        logging.error("Calculate AUPRc for " + self.chromosome)

        # Calculate AUPRc
        self.AUPRC = metrics.auc(y=self.precision[:-1], x=self.recall[:-1])

        self.PR_CURVE_DF["AUPRC"] = self.AUPRC

        # Calculate the total gold standard bins
        logging.error("Calculate Total GoldStandard Bins")

        self.PR_CURVE_DF["Total_GoldStandard_Bins"] = len(np.argwhere(self.goldstandard_array == True))

        # Find the number of non-blacklisted bins in chr of interest
        rand_bins = len(np.argwhere(self.blacklist_mask == True))

        # Random Precision
        self.PR_CURVE_DF['Random_AUPRC'] = self.PR_CURVE_DF['Total_GoldStandard_Bins'] / rand_bins

        # Log2FC
        self.PR_CURVE_DF['log2FC_AUPRC_Random_AUPRC'] = np.log2(self.PR_CURVE_DF["AUPRC"] / self.PR_CURVE_DF["Random_AUPRC"])

        logging.error("Write results for " + self.chromosome)

        # Write the AUPRC stats to a dataframe
        self.PR_CURVE_DF.to_csv(self.results_location, sep="\t", header=True, index=False)

    def __plot(self, cmap="viridis"):
        points = np.array([self.recall, self.precision]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        fig, axs = plt.subplots(1, figsize=(5, 4), dpi=150)

        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(0, 1)

        lc = LineCollection(segments, cmap=cmap, norm=norm)
        # Set the values used for colormapping
        lc.set_array(self.thresholds)
        lc.set_linewidth(5)
        line = axs.add_collection(lc)
        fig.colorbar(line)
        plt.grid()
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.ylabel("Precision")
        plt.xlabel("Recall")

        plt.savefig(remove_tags(self.results_location, ".tsv") + ".png")
