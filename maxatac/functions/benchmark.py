import logging
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from maxatac.utilities.genome_tools import load_bigwig

def benchmark_predictions(
    prediction,
    goldstandard,
    bin_size,
    chrom,  
    results_location# (chrom_name)
):
    with load_bigwig(prediction) as prediction_stream, load_bigwig(goldstandard) as goldstandard_stream:
        chrom_len = prediction_stream.chromosomes(chrom)
        
        bin_count = int(int(chrom_len) / int(bin_size))  # need to floor the number

        logging.error(
            "Prediction file:" + prediction +
            "\n  Gold standard file: " + goldstandard +
            "\n  Chromosome: " + chrom +
            "\n  Binning: " + str(bin_count) + " bins * " + str(bin_size) + " bp"
            "\n  Results location: " + results_location
        )
    
        prediction_chrom_data = np.nan_to_num(
            np.array(
                prediction_stream.stats(
                    chrom,
                    0,
                    chrom_len,
                    type="max",
                    nBins=bin_count,
                    exact=True
                ),
                dtype=float  # need it to have NaN instead of None
            )
        )

        logging.error("Import Gold Standard Array")

        goldstandard_chrom_data = np.nan_to_num(
            np.array(
                goldstandard_stream.stats(
                    chrom,
                    0,
                    chrom_len,
                    type="mean",
                    nBins=bin_count,
                    exact=True
                ),
                dtype=float  # need it to have NaN instead of None
            )
        ) > 0  # to convert to boolean array
        
        logging.error("Calculate Precision and Recall")

        # Calculate the precision and recall of both numpy arrays
        precision, recall, thresholds = precision_recall_curve(goldstandard_chrom_data, prediction_chrom_data)

        logging.error("Results Cleanup and Writing")
        # Create a dataframe from the results
        PR_CURVE_DF = pd.DataFrame({'Precision':precision, 'Recall':recall})

        # The sklearn documentation states they add points to make sure precision and recall are at 1. We remove this point 
        point_to_drop = PR_CURVE_DF[(PR_CURVE_DF["Recall"] == 1.0)].index

        # This will drop the point from the dataframe
        PR_CURVE_DF.drop(point_to_drop, inplace=True)

        try:
            PR_CURVE_DF["AUPR"] = metrics.auc(y=precision, x=recall)

        except:
            PR_CURVE_DF["AUPR"] = 0

        PR_CURVE_DF.to_csv(results_location, sep="\t", header=True, index=False)

def run_benchmarking(args):
    results_filename = args.output + "/" + args.prefix + "_" + str(args.bin) + "bp_PRC.tsv"

    logging.error(
        "Benchmarking" +
        "\n  Prediction file:" + args.prediction +
        "\n  Gold standard file: " + args.goldstandard +
        "\n  Bin size: " + str(args.bin) +
        "\n  All chromosomes: " + args.chromosomes +
        "\n  Logging level: " + logging.getLevelName(args.loglevel) +
        "\n  Output directory: " + results_filename
    )
    
    benchmark_predictions(
                args.prediction,
                args.goldstandard,
                args.bin,
                args.chromosomes,
                results_filename
    )