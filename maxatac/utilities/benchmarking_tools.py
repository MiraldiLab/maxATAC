import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from maxatac.utilities.genome_tools import load_bigwig, chromosome_blacklist_mask, chromosome_whitelist_mask, import_prediction_array_fn, import_quant_goldstandard_array_fn
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, r2_score, mean_absolute_error
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from maxatac.utilities.system_tools import remove_tags
import pybedtools
from multiprocessing import Pool
import multiprocessing
import tqdm


def Precision_for_Recall(df, percent_recall):
    percent_recall = percent_recall
    upper_lim_recall = df.iloc[(df['Recall'] - percent_recall).abs().argsort()[:2]].Recall.tolist()[0]
    lower_lim_recall = df.iloc[(df['Recall'] - percent_recall).abs().argsort()[:2]].Recall.tolist()[1]
    upper_lim_precision = df.iloc[(df['Recall'] - percent_recall).abs().argsort()[:2]].Precision.tolist()[0]
    lower_lim_precision = df.iloc[(df['Recall'] - percent_recall).abs().argsort()[:2]].Precision.tolist()[1]
    val = (upper_lim_precision * abs(percent_recall - upper_lim_recall) + lower_lim_precision * abs(
        percent_recall - lower_lim_recall)) / 2
    sp_precision = lower_lim_precision + val
    return sp_precision

def calculate_sse(vector1, vector2):
    """
    Calculates the sum of squared errors (SSE) between two NumPy vectors.

    Args:
        vector1 (np.ndarray): The first vector.
        vector2 (np.ndarray): The second vector.

    Returns:
        float: The sum of squared errors.
    """
    if vector1.shape != vector2.shape:
        raise ValueError("Vectors must have the same shape.")

    squared_differences = (vector1 - vector2) ** 2
    sse = np.sum(squared_differences)
    return sse

class calculate_R2_pearson_spearman(object):
    """
    Calculate the R2, Pearson, and Spearman Correlation for Quantitative Predictions
    :param prediction_bw: The input prediction bigwig file
    :param gold_standard_bw: The input gold standard file
    :param chromosome: The chromosome to limit the analysis to
    :param results_location: The location to write the results to
    :param blacklist_bw: The blacklist mask that is used to remove bins overlapping blacklist regions

    :return: Writes a TSV for the P/R curve
    """

    def __init__(self,
                 prediction_bw,
                 goldstandard_bw,
                 quant_goldstandard_bw,
                 chromosome,
                 bin_size,
                 agg_function,
                 results_location,
                 blacklist_bw,
                 whitelist_bw,
                 quant_gs_null,
                 alternative_prediction_bw=None,
                 prediction_combine_operation="mean"
                 ):

        """
        Initialize all input values as part of the Class
        """
        self.results_location = results_location

        self.prediction_stream = load_bigwig(prediction_bw)
        self.alternative_prediction_stream = load_bigwig(alternative_prediction_bw) if alternative_prediction_bw else None
        self.goldstandard_stream = load_bigwig(goldstandard_bw)
        self.quant_goldstandard_stream = load_bigwig(quant_goldstandard_bw)
        self.quant_gs_null_stream = load_bigwig(quant_gs_null)

        self.chromosome = chromosome
        self.chromosome_length = self.goldstandard_stream.chroms(self.chromosome)

        self.bin_count = int(int(self.chromosome_length) / int(bin_size))  # need to floor the number
        self.bin_size = bin_size
        self.agg_function = agg_function
        self.quant_gs_null = quant_gs_null # change to quant_null_model
        self.prediction_combine_operation = prediction_combine_operation

        # This has been modified
        self.blacklist_mask = chromosome_blacklist_mask(blacklist_bw,
                                                        self.chromosome,
                                                        self.chromosome_length,
                                                        self.bin_count)

        if whitelist_bw:
            self.blacklist_mask = np.logical_and(
                self.blacklist_mask,
                chromosome_whitelist_mask(whitelist_bw,
                                          self.chromosome,
                                          self.chromosome_length,
                                          self.bin_count)
            )

        '''self.blacklist_mask = chromosome_blacklist_mask(blacklist_bw,
                                                        self.chromosome,
                                                        self.chromosome_length,
                                                        self.chromosome_length #self.bin_count #nBins= bin_size
                                                        )'''

        # Call on the def in the class object to do the calculations
        self.__import_prediction_array__()
        self.__import_goldstandard_array__()
        self.__import_quant_goldstandard_array__()
        self.__import_quant_goldstandard_null_array__()
        self.__R2_Sp_P__()
        self.__plot__()

    def __import_prediction_array__(self):
        """
        Import the chromosome signal from the predictions bigwig file and convert to a numpy array.
        """
        logging.info("Import Predictions Array")

        # Get the bin stats from the prediction array

        self.prediction_array = import_prediction_array_fn(self.prediction_stream,
                                                         self.chromosome,
                                                         self.chromosome_length,
                                                         self.agg_function,
                                                         self.bin_count,
                                                         alternative_prediction_stream=self.alternative_prediction_stream,
                                                         combine_operation=self.prediction_combine_operation)



    def __import_goldstandard_array__(self):
        """
        Import the chromosome signal from the gold standard bigwig file and convert to a numpy array.
        """

        logging.info("Import Gold Standard Array")
        # prediction_chromosome_data = np.round(prediction_chromosome_data, round_predictions)

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
                                                ) # Commented out to keep values non-boolean:  > 0  # to convert to boolean array


    def __import_quant_goldstandard_array__(self):
        """
        Import the chromosome signal from the gold standard bigwig file and convert to a numpy array.
        """

        logging.info("Import Quantitative Gold Standard Array")
        # prediction_chromosome_data = np.round(prediction_chromosome_data, round_predictions)

        # Get the bin stats from the gold standard array

        self.quant_goldstandard_array = np.nan_to_num(np.array(self.quant_goldstandard_stream.stats(self.chromosome,
                                                                                        0,
                                                                                        self.chromosome_length,
                                                                                        type=self.agg_function,
                                                                                        nBins=self.bin_count,
                                                                                        exact=True
                                                                                        ),
                                                         dtype=float  # need it to have NaN instead of None
                                                         )
                                                ) # Commented out to keep values non-boolean:  > 0  # to convert to boolean array

    def __import_quant_goldstandard_null_array__(self):
        """
        Import the chromosome signal from the gold standard null model bigwig file and convert to a numpy array.
        """

        logging.info("Import Quantitative Gold Standard Null Model as Array")
        # prediction_chromosome_data = np.round(prediction_chromosome_data, round_predictions)

        # Get the bin stats

        self.quant_goldstandard_null_array = np.nan_to_num(np.array(self.quant_gs_null_stream.stats(self.chromosome,
                                                                                                    0,
                                                                                                    self.chromosome_length,
                                                                                                    type=self.agg_function,
                                                                                                    nBins=self.bin_count,
                                                                                                    exact=True
                                                                                                    ),
                                                               dtype=float  # need it to have NaN instead of None
                                                               )
                                                      )  # Commented out to keep values non-boolean:  > 0  # to convert to boolean array

    def __R2_Sp_P__(self):
        """
        Calculate the R2, Pearson, and Spearman Correlation for Quantitative Predictions
        """
        ### dfdf = pd.read_csv(self.pred_gs_meta, sep='\t')
        ### dim = dfdf.shape[0]

        blacklist_mask = self.blacklist_mask  # Assuming this is a member variable
        chromosome = self.chromosome  # Assuming this is a member variable
        chromosome_length = self.chromosome_length  # Assuming this is a member variable
        agg_function = self.agg_function  # Assuming this is a member variable
        bin_count = self.bin_count  # Assuming this is a member variable


        logging.info("Calculate R2_pred")
        SSE_pred = calculate_sse(
            self.quant_goldstandard_array[self.blacklist_mask],
            self.prediction_array[self.blacklist_mask])
        SSE_null = calculate_sse(
            self.quant_goldstandard_array[self.blacklist_mask],
            self.quant_goldstandard_null_array[self.blacklist_mask])
        R2_pred = 1 - SSE_pred / SSE_null


        logging.info("Calculate Pearson Correlation")
        pearson_score, pearson_pval = pearsonr(
            self.quant_goldstandard_array[self.blacklist_mask],
            self.prediction_array[self.blacklist_mask]
            )

        logging.info("Calculate Spearman Correlation")
        spearman_score, spearman_pval = spearmanr(
            self.quant_goldstandard_array[self.blacklist_mask],
            self.prediction_array[self.blacklist_mask]
            )

        logging.info("Calculate MAE")
        mae_val = mean_absolute_error(
            self.quant_goldstandard_array[self.blacklist_mask],
            self.prediction_array[self.blacklist_mask]
        )



        R2_Sp_P_df = pd.DataFrame([[mae_val, R2_pred, pearson_score, pearson_pval, spearman_score, spearman_pval]],
                                  columns=['MAE', 'R2_pred', 'pearson', 'pearson_pval', 'spearman', 'spearman_pval'])

        R2_Sp_P_df.to_csv(self.results_location, sep='\t', index=None, float_format='%.6e')

    def __plot__(self):

        # genomic coordinates
        chr_region = pybedtools.BedTool(f"{self.chromosome}\t0\t{self.chromosome_length}\n", from_string=True)
        chr_df = pybedtools.BedTool().window_maker(b=chr_region, w=self.bin_size).to_dataframe()
        if chr_df.tail(1).end.to_numpy()[0] - chr_df.tail(1).start.to_numpy()[0] != self.bin_size:
            chr_df = chr_df.drop(chr_df.index[-1])
        else:
            pass
        chr_df_filt = chr_df.loc[self.blacklist_mask]

        y_pred= self.prediction_array[self.blacklist_mask]
        y_obs= self.quant_goldstandard_array[self.blacklist_mask]


        pred_obs_data = {'y_pred': y_pred, 'y_obs': y_obs}
        pred_obs_df = pd.DataFrame(pred_obs_data)

        # Reset index
        chr_df_filt.reset_index(drop=True, inplace=True)
        pred_obs_df.reset_index(drop=True, inplace=True)

        plot_df = pd.concat([chr_df_filt, pred_obs_df], axis=1)

        logging.info("Creating Scatterplot")

        # plotting figure
        fig, ax = plt.subplots()
        x = plot_df.y_obs
        y = plot_df.y_pred

        # Fit a line of best fit
        (m, b), (SSE,), *_ = np.polyfit(x, y, deg=1, full=True)
        # set y-intercept = 0
        b=0
        from matplotlib.ticker import MaxNLocator
        import matplotlib.ticker as ticker

        # Generate values for the line of best fit
        xseq = np.linspace(min(x) - 1, max(x) + 1, num=100)
        ax.plot(xseq, m * xseq + b, color='r', lw=2.5, label=f'Best Fit: y = {m:.2f}x + {b:.2f}\nSSE = {SSE:.2f}')

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # line y=x
        plt.plot(xseq, xseq, label='y=x', color='lightgray')

        fit_y = m * xseq + b
        line_y = xseq

        # Calculate R2_(y=x) (R2 between the line of best and the line y=x y-values)
        R2_yisx = r2_score(line_y, fit_y)

        ax.scatter(x, y, s=60, alpha=0.0020, label='Data points')

        plt.title("Observed vs Predicted Scatter" + '\n \n')
        plt.xlabel("Observed", size=20)
        plt.ylabel("Predicted", size=20)

        plt.grid(True, linestyle='-', color='gray', alpha=0.5)
        plt.xticks(size=18)
        plt.yticks(size=18)

        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
        plt.grid(which='major', linestyle='-', linewidth='1', color='grey', alpha=.6)


        plot_location='_'.join([self.results_location.split(".")[0], "scatterPlot.png"])

        logging.info("Saving Scatterplot")

        fig.savefig(plot_location,
            bbox_inches="tight"
        )

        logging.info("Saving Scatterplot DF")

        plot_df_location = '_'.join([self.results_location.split(".")[0], "scatterPlot_df.tsv"])

        plot_df.to_csv(plot_df_location, sep='\t', index=None)

        R2_yisx_Slope_df = pd.DataFrame([[R2_yisx, m]],
                                  columns=['R2_yisx', 'Slope'])
        R2_yisx_Slope_df_location = "_".join(["_".join(self.results_location.split("_")[:-3]), "R2_yisx_Slope_df.tsv"])

        logging.info("Saving R2_yisx")
        R2_yisx_Slope_df.to_csv(R2_yisx_Slope_df_location, sep='\t', index=None, float_format='%.6e')

# masking v1_peak_gs array
def peak_gs_array_to_member_array(array):
  member_array=[]
  positive_count=0
  negative_count=0
  for i in range(len(array)):
    if i==0:
      if array[i]==True:
        positive_count+=1
        member_array.append(positive_count)
      if array[i]==False:
        negative_count-=1
        member_array.append(negative_count)
    else:
      if (array[i])==True:
        if array[i-1]==True:
          # positive_count+=1
          member_array.append(positive_count) # same peak
        else:
          positive_count+=1
          member_array.append(positive_count) # new peak
      else:
        negative_count-=1
        member_array.append(negative_count)
  member_array=np.asarray(member_array)
  return member_array

def cum_unique_count(arr):
    seen_pos = set()
    seen_neg = set()
    tp_array = []
    fp_array = []
    for x in tqdm.tqdm(arr):
        if x > 0:
            seen_pos.add(x)
        elif x < 0:
            seen_neg.add(x)
        tp_array.append(len(seen_pos))
        fp_array.append(len(seen_neg))
    return tp_array, fp_array

def precision_recall_curve_peak_based(y_member, y_score):
    # Convert to boolean
    #y_true = np.asarray(y_true) #  == pos_label # no need to convert to binary label, e.g., for quant results
    y_score = np.asarray(y_score)
    y_member = np.asarray(y_member)
    # membership of each 32bp bin, if certain 32bp bins belong to the same peak,
    # they would have same positive value in the member array,
    # if negative prediction, each has a unique negative value
    #
    # Sort by score descending
    order = np.argsort(y_score)[::-1]
    y_score = y_score[order]
    #y_true = y_true[order]
    y_member = y_member[order]
    #
    #
    # Count positives
    # total_positives = np.sum(y_true)
    # Now count with unique member
    total_positives = len(np.unique(y_member[np.where(y_member>0)]))  # how many unique positive membership ids (# of unique peaks) in the entire input array
    total_negatives= len(np.unique(y_member[np.where(y_member<0)]))  # how many unique negative labeled membership ids in the entire input array
    #
    # Cumulative TP and FP
    tp,fp=cum_unique_count(y_member)
    tp=np.asarray(tp)
    fp=np.asarray(fp)
    #tp = np.cumsum(y_true)
    #fp = np.cumsum(~y_true)
    #
    # Find indices where score changes
    distinct_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_indices, len(y_member) - 1]
    #
    # Thresholds
    thresholds = y_score[threshold_idxs]
    #
    # Precision and recall
    precision = tp[threshold_idxs] / (tp[threshold_idxs] + fp[threshold_idxs])
    recall = tp[threshold_idxs] / total_positives
    #
    # Add endpoint
    precision = np.r_[precision, 1.0]
    recall = np.r_[recall, 0.0]
    #
    return precision, recall, thresholds

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
                 whitelist_bw,
                 chromosome,
                 bin_size,
                 agg_function,
                 agg_threshold,
                 results_location,
                 round_predictions,
                 plot=False,
                 peak_based=False,
                 alternative_prediction_bw=None,
                 prediction_combine_operation="mean"
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
        self.alternative_prediction_stream = load_bigwig(alternative_prediction_bw) if alternative_prediction_bw else None
        self.goldstandard_stream = load_bigwig(goldstandard_bw)

        self.chromosome = chromosome
        self.chromosome_length = self.goldstandard_stream.chroms(self.chromosome)

        self.bin_count = int(int(self.chromosome_length) / int(bin_size))  # need to floor the number
        self.bin_size = bin_size

        self.agg_function = agg_function
        self.prediction_combine_operation = prediction_combine_operation
        self.agg_threshold = agg_threshold

        self.blacklist_mask = chromosome_blacklist_mask(blacklist_bw,
                                                        self.chromosome,
                                                        self.chromosome_length,
                                                        self.bin_count)

        if whitelist_bw:
            self.blacklist_mask = np.logical_and(
                self.blacklist_mask,
                chromosome_whitelist_mask(whitelist_bw,
                                          self.chromosome,
                                          self.chromosome_length,
                                          self.bin_count)
            )

        self.peak_based=peak_based
        self.__import_prediction_array__(round_prediction=round_predictions)
        self.__import_goldstandard_array__()
        self.__AUPRC__()

        if plot:
            logging.info("Plotting AUPRC Curves")
            self.__plot()

    def __import_prediction_array__(self, round_prediction=6):
        """
        Import the chromosome signal from the predictions bigwig file and convert to a numpy array.

        :param round_prediction: The number of floating places to round the signal to
        :return: prediction_array: A np.array that has values binned according to bin_count and aggregated according
        to agg_function
        """
        logging.info("Import Predictions Array")

        # Get the bin stats from the prediction array
        self.prediction_array = import_prediction_array_fn(self.prediction_stream,
                                                         self.chromosome,
                                                         self.chromosome_length,
                                                         self.agg_function,
                                                         self.bin_count,
                                                         alternative_prediction_stream=self.alternative_prediction_stream,
                                                         combine_operation=self.prediction_combine_operation)

        # Under "sum" the bin value is a base-pair total; divide by bin_size to stay on the
        # same per-base scale as max/mean.
        if self.agg_function == "sum":
            self.prediction_array = self.prediction_array / self.bin_size

        self.prediction_array = np.round(self.prediction_array, round_prediction)

    def __import_goldstandard_array__(self):
        """
        Import the chromosome signal from the gold standard bigwig file and convert to a numpy array with True/False
        entries.

        :return: goldstandard_array: A np.array has values binned according to bin_count and aggregated according to
        agg_function. random_precision: The random precision of the model based on # of True bins/ # of genomic bins
        """
        logging.info("Import Gold Standard Array")

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
                                                )

        if self.agg_function == "sum":
            self.goldstandard_array = np.where((self.goldstandard_array / self.bin_size) >= self.agg_threshold,
                                               1.0,
                                               0.0)
        else:
            self.goldstandard_array = self.goldstandard_array > 0

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
        logging.info("Calculate precision-recall curve for " + self.chromosome)

        if self.peak_based==False:
            self.precision, self.recall, self.thresholds = precision_recall_curve(
                self.goldstandard_array[self.blacklist_mask],
                self.prediction_array[self.blacklist_mask]
            )
        else:
            # generate member array from gs
            self.precision, self.recall, self.thresholds = precision_recall_curve_peak_based(
                peak_gs_array_to_member_array(self.goldstandard_array[self.blacklist_mask]),
                self.prediction_array[self.blacklist_mask]
            )
        logging.info("Making DataFrame from results")

        # Create a dataframe from the results
        # Issue 54:
        # The sklearn package will add a point at precision=1 and recall=0
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
        # remove the last point of the array which corresponds to this extra point
        self.PR_CURVE_DF = pd.DataFrame(
            {'Precision': self.precision[:-1], 'Recall': self.recall[:-1], "Threshold": self.thresholds})

        logging.info("Calculate AUPRc for " + self.chromosome)

        # Calculate AUPRc
        self.AUPRC = metrics.auc(y=self.precision[:-1], x=self.recall[:-1])

        self.PR_CURVE_DF["AUPRC"] = self.AUPRC

        if self.peak_based==False:
            # Calculate the total gold standard bins
            logging.info("Calculate Total GoldStandard Bins")

            self.PR_CURVE_DF["Total_GoldStandard_Bins"] = len(np.argwhere(self.goldstandard_array == True))

            # Find the number of non-blacklisted bins in chr of interest
            rand_bins = len(np.argwhere(self.blacklist_mask == True))

            # Random Precision
            self.PR_CURVE_DF['Random_AUPRC'] = self.PR_CURVE_DF['Total_GoldStandard_Bins'] / rand_bins

            # Log2FC
            self.PR_CURVE_DF['log2FC_AUPRC_Random_AUPRC'] = np.log2(self.PR_CURVE_DF["AUPRC"] / self.PR_CURVE_DF["Random_AUPRC"])

        # Precision at 10% Recall
        self.PR_CURVE_DF['Precision_at_10_Percent_Recall']=Precision_for_Recall(self.PR_CURVE_DF, 0.1)

        logging.info("Write results for " + self.chromosome)

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
