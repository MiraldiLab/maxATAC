import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from maxatac.utilities.genome_tools import load_bigwig, chromosome_blacklist_mask
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy import stats
from maxatac.utilities.system_tools import remove_tags


def calculate_R2_pearson_spearman(prediction,
                                  gold_standard,
                                  chromosome,
                                  results_location,
                                  blacklist_mask
                                  ):
    """
    Calculate the R2, Pearson, and Spearman Correlation for Quantitative Preidcitions

    :param prediction: The input prediction bigwig file
    :param gold_standard: The input gold standard file
    :param chromosome: The chromosome to limit the analysis to
    :param results_location: The location to write the results to
    :param blacklist_mask: The blacklist mask that is used to remove bins overlapping blacklist regions

    :return: Writes a TSV for the P/R curve
    """
    with load_bigwig(prediction) as prediction_stream, load_bigwig(gold_standard) as goldstandard_stream:
        # Get the end of the chromosome
        chromosome_length = prediction_stream.chroms(chromosome)

        logging.error("Import Predictions Array for Quantitative Predictions")

        # Get the bin stats from the prediction array
        prediction_chromosome_data = np.nan_to_num(
            prediction_stream.values(
                chromosome,
                0,
                chromosome_length)
        )

        logging.error("Import Gold Standard Array")
        # prediction_chromosome_data = np.round(prediction_chromosome_data, round_predictions)

        # Get the bin stats from the gold standard array

        gold_standard_chromosome_data = np.nan_to_num(
            goldstandard_stream.values(
                chromosome,
                0,
                chromosome_length)
        )

        logging.error("Calculate R2")
        R2_score = r2_score(gold_standard_chromosome_data[blacklist_mask],
                            prediction_chromosome_data[blacklist_mask])

        logging.error("Calculate Pearson Correlation")

        pearson_score, pearson_pval = pearsonr(gold_standard_chromosome_data[blacklist_mask],
                                               prediction_chromosome_data[blacklist_mask])

        logging.error("Calculate Spearman Correlation")

        spearman_score, spearman_pval = stats.spearmanr(gold_standard_chromosome_data[blacklist_mask],
                                                        prediction_chromosome_data[blacklist_mask])

        R2_Sp_P_df = pd.DataFrame([[R2_score, pearson_score, pearson_pval, spearman_score, spearman_pval]],
                                  columns=['R2', 'pearson', 'pearson_pval', 'spearman', 'spearman_pval'])

        R2_Sp_P_df.to_csv(results_location, sep='\t', index=None)


class ChromosomeAUPRC(object):
    """
    Benchmark maxATAC binary predictions against a gold standard using AUPRC. You can also input quantitative
    predictions, but they will ranked by significance.

    During initializiation the following steps will be performed:

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
                 round_predictions):
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
        self.agg_function = agg_function

        self.blacklist_mask = chromosome_blacklist_mask(blacklist_bw, self.chromosome, self.chromosome_length,
                                                        self.bin_count)

        self.__import_prediction_array__(round_prediction=round_predictions)
        self.__import_goldstandard_array__()

        self.__AUPRC__()

        self.__plot()

    def __import_prediction_array__(self, round_prediction=6):
        """
        Import the chromosome signal from the predictions bigwig file and convert to a numpy array.

        :param round_prediction: The number of floating places to round the signal to
        :return: prediction_array: A np.array that has values binned according to bin_count and aggregated according to agg_function
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
                                                       ))

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
        self.PR_CURVE_DF = pd.DataFrame(
            {'Precision': self.precision, 'Recall': self.recall, "Threshold": np.insert(self.thresholds, 0, 0)})

        logging.error("Calculate AUPRc for " + self.chromosome)
        # Calculate AUPRc
        self.AUPRC = metrics.auc(y=self.precision[:-1], x=self.recall[:-1])

        self.PR_CURVE_DF["AUPRC"] = self.AUPRC

        #logging.error("Calculate AUC for each threshold")
        # Calculate AUC for each threshold
        #self.PR_CURVE_DF["AUC"] = self.PR_CURVE_DF["Threshold"].apply(lambda x: self.__calculate_AUC_per_rank__(x))

        # Calculate the total number of predictions for each threshold
        #self.PR_CURVE_DF["Number_of_Predictions"] = self.PR_CURVE_DF["Threshold"].apply(
        #    lambda x: self.__get_bin_count__(x))

        # Calculate the total gold standard bins
        logging.error("Calculate Total GoldStandard Bins")
        
        self.PR_CURVE_DF["Total_GoldStandard_Bins"] = len(np.argwhere(self.goldstandard_array == True))

        # Calculate the true positives at each cutoff
        #self.PR_CURVE_DF["True_positive"] = self.PR_CURVE_DF["Threshold"].apply(
        #    lambda x: self.__get_true_positives__(x))

        # Calculate the false positive at each cutoff
        #self.PR_CURVE_DF["False_positive"] = self.PR_CURVE_DF["Threshold"].apply(
        #    lambda x: self.__get_false_positives__(x))

        # Calculate the false negative at each cutoff
        #self.PR_CURVE_DF["False_negative"] = self.PR_CURVE_DF["Total_GoldStandard_Bins"] - self.PR_CURVE_DF[
        #    "True_positive"]

        logging.error("Write results for " + self.chromosome)

        # Write the AUPRC stats to a dataframe
        self.PR_CURVE_DF.to_csv(self.results_location, sep="\t", header=True, index=False)

    def __plot(self, cmap="viridis"):
        points = np.array([self.recall, self.precision]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        fig, axs = plt.subplots(1, figsize=(5, 4), dpi=150)

        # plt.plot(PR_CURVE_DF.Recall, PR_CURVE_DF.Threshold)
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
