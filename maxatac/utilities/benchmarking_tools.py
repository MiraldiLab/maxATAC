import logging
import numpy as np
import pandas as pd

from maxatac.utilities.genome_tools import load_bigwig
from sklearn import metrics
from sklearn.metrics import precision_recall_curve


def get_blacklist_mask(blacklist,
                       chromosome,
                       bin_size
                       ):
    """
    Get a numpy array that will allow you to select non-blacklisted regions for benchmarking. This currently only works
    for a single chromosome.

    :param blacklist: Input blacklist bigwig file
    :param chromosome:  Chromosome that you are benchmarking on
    :param bin_size: Size of the bins for comparison

    :return : An array that has True in positions that are not blacklisted
    """
    with load_bigwig(blacklist) as blacklist_stream:
        # Get the chromosome ends
        chromosome_length = blacklist_stream.chroms(chromosome)

        # Get the number of bins for the genome
        bin_count = int(int(chromosome_length) / int(bin_size))  # need to floor the number

        # TODO rewrite to maybe have pybigwig.stats calculate the bin number based on span and width
        # Get the blacklist values as an array
        return np.array(blacklist_stream.stats(chromosome, 0, chromosome_length, type="max", nBins=bin_count),
                        dtype=float  # need it to have NaN instead of None
                        ) != 1  # Convert to boolean array, select areas that are not 1


def calculate_predictions_AUPR(prediction,
                               gold_standard,
                               bin_size,
                               chromosome,
                               results_location,
                               blacklist_mask,
                               agg_function,
                               round_predictions
                               ):
    """
    Calculate the precision and recall of a ChIP-seq gold standard

    :param prediction: The input prediction bigwig file
    :param gold_standard: The input gold standard file
    :param bin_size: The bin size to compare the predictions at
    :param chromosome: The chromosome to limit the analysis to
    :param results_location: The location to write the results to
    :param blacklist_mask: The blacklist mask that is used to remove bins overlapping blacklist regions
    :param round_predictions: Round the prediction values to this many floating points
    :param agg_function: The function to use to aggregate scores in bins

    :return: Writes a TSV for the P/R curve
    """
    # TODO There is a way to use pybigwig to bin the data with specific intervals and steps. We need to use this
    with load_bigwig(prediction) as prediction_stream, load_bigwig(gold_standard) as goldstandard_stream:
        # Get the end of the chromosome
        chromosome_length = prediction_stream.chroms(chromosome)

        # Get the number of bins to use for the chromosome
        bin_count = int(int(chromosome_length) / int(bin_size))  # need to floor the number

        logging.error("Import Predictions Array")

        # Get the bin stats from the prediction array
        prediction_chromosome_data = np.nan_to_num(
            np.array(
                prediction_stream.stats(
                    chromosome,
                    0,
                    chromosome_length,
                    type=agg_function,
                    nBins=bin_count,
                    exact=True
                ),
                dtype=float  # need it to have NaN instead of None
            )
        )

        logging.error("Import Gold Standard Array")

        prediction_chromosome_data = np.round(prediction_chromosome_data, round_predictions)

        # Get the bin stats from the gold standard array
        gold_standard_chromosome_data = np.nan_to_num(np.array(goldstandard_stream.stats(chromosome,
                                                                                         0,
                                                                                         chromosome_length,
                                                                                         type="mean",
                                                                                         nBins=bin_count,
                                                                                         exact=True
                                                                                         ),
                                                               dtype=float  # need it to have NaN instead of None
                                                               )
                                                      ) > 0  # to convert to boolean array

        logging.error("Calculate Precision and Recall")

        # Calculate the precision and recall of both numpy arrays
        precision, recall, thresholds = precision_recall_curve(gold_standard_chromosome_data[blacklist_mask],
                                                               prediction_chromosome_data[blacklist_mask])

        logging.error("Results Cleanup and Writing")

        # Create a dataframe from the results
        PR_CURVE_DF = pd.DataFrame({'Precision': precision, 'Recall': recall})

        # TODO Make sure that the calculation of AUPR is correct
        # The sklearn documentation states they add points to make sure precision and recall are at 1. We remove this
        # point
        point_to_drop = PR_CURVE_DF[(PR_CURVE_DF["Recall"] == 1.0)].index

        # This will drop the point from the dataframe
        PR_CURVE_DF.drop(point_to_drop, inplace=True)

        # If there are no prediction there will be an error when you try to calculate the AUPR so set to 0
        try:
            PR_CURVE_DF["AUPR"] = metrics.auc(y=precision, x=recall)

        except:
            PR_CURVE_DF["AUPR"] = 0

        # TODO Draw a line from the first point to 0.

        # Write the precision recall curve to a file
        PR_CURVE_DF.to_csv(results_location, sep="\t", header=True, index=False)
