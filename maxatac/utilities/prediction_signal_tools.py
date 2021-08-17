import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from maxatac.utilities.genome_tools import load_bigwig, chromosome_blacklist_mask, load_2bit
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy import stats
from maxatac.utilities.system_tools import remove_tags

class Predicion_Signal(object):
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
                 sequence,
                 chromosome,
                 bin_size,
                 agg_function,
                 results_location,
                 round_predictions):

        """
        :param prediction_bw: Path to bigwig file containing maxATAC predictions
        :param hg38_2bit: Path to sequence
        :param blacklist_bw: Path to blacklist bigwig file
        :param chromosome: Chromosome to benchmark
        :param bin_size: Resolution to bin the results to
        :param agg_function: Method to use to aggregate multiple signals in the same bin
        """
        self.results_location = results_location

        self.prediction_stream = load_bigwig(prediction_bw)

        self.chromosome = chromosome

        sequence = load_2bit(sequence)
        self.chromosome_length = sequence.chroms(self.chromosome)

        self.bin_count = int(int(self.chromosome_length) / int(bin_size))  # need to floor the number
        self.bin_size = bin_size

        self.agg_function = agg_function


        self.__import_prediction_array__(round_prediction=round_predictions)


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

        logging.error("Creating Prediction Values")

        prediction_val_df= pd.DataFrame({"chr": self.chromosome,
                                         "start": np.arange(0, self.bin_count * self.bin_size, self.bin_size),
                                         "stop": np.arange(self.bin_size, self.bin_count * self.bin_size + self.bin_size, self.bin_size),
                                         "count": self.prediction_array
                                         })

        self.results_location = '.'.join(['_'.join([self.results_location.split(".")[0][:-4], 'prediction_value']), 'tsv'])


        prediction_val_df.to_csv(self.results_location, sep='\t', index=False)