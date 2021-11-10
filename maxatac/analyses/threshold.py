import numpy as np
import pandas as pd
import os
from maxatac.utilities.genome_tools import build_chrom_sizes_dict, get_bigwig_stats
from maxatac.utilities.system_tools import get_dir
from maxatac.utilities.threshold_tools import import_blacklist_mask, import_GoldStandard_array, calculate_AUC_per_rank
from sklearn.metrics import precision_recall_curve
from sklearn import metrics


def run_thresholding(args):
    """
    :param args:
    :return:
    """
    # Make the output directory
    output_dir = get_dir(args.output_dir)

    chromosome_sizes_dictionary = build_chrom_sizes_dict(args.chromosomes, args.chrom_sizes)

    meta_DF = pd.read_table(args.meta_file)

    training_data_dict = pd.Series(meta_DF["Binding_File"].values,index=meta_DF["Prediction"]).to_dict()

    results_filename = os.path.join(output_dir,
                                    args.prefix + "_" + args.chromosomes[0] + "_" + str(args.bin_size) + "bp_PRC.tsv")

    # Loop through the chromosomes and average the values across files
    for chrom_name, chrom_length in chromosome_sizes_dictionary.items():
        bin_count = int(int(chrom_length) / int(args.bin_size))  # need to floor the number

        blacklist_mask = import_blacklist_mask(args.blacklist, chrom_name, chrom_length, bin_count)

        blacklist = np.repeat(blacklist_mask, len(training_data_dict.keys()))

        predictions = np.empty(1)

        gold_standard = np.empty(1)

        print(chrom_length)
        print(bin_count)
        # Loop through the bigwig files and get the values and add them to the array.
        for bigwig_file in training_data_dict.keys():
            chrom_vals = get_bigwig_stats(bigwig_file, chrom_name, chrom_length, bin_count)

            print(training_data_dict[bigwig_file])
            goldstandard_array = import_GoldStandard_array(training_data_dict[bigwig_file], chrom_name, chrom_length, bin_count)

            predictions = np.concatenate([predictions, chrom_vals])
            gold_standard = np.concatenate([gold_standard, goldstandard_array])

        predictions = np.delete(predictions, 0)
        gold_standard = np.delete(gold_standard, 0)

        precision, recall, thresholds = precision_recall_curve(
            gold_standard[blacklist],
            predictions[blacklist])

        # Create a dataframe from the results
        PR_CURVE_DF = pd.DataFrame(
            {'Precision': precision, 'Recall': recall, "Threshold": np.insert(thresholds, 0, 0)})

        # Calculate AUPRc
        AUPRC = metrics.auc(y=precision[:-1], x=recall[:-1])

        PR_CURVE_DF["AUC"] = PR_CURVE_DF["Threshold"].apply(lambda x: calculate_AUC_per_rank(PR_CURVE_DF, x))

        PR_CURVE_DF.to_csv(results_filename, sep="\t", header=True, index=False)