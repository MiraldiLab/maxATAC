import logging
import os

from maxatac.utilities.system_tools import get_dir, Mute

with Mute():
    from maxatac.utilities.benchmarking_tools import calculate_predictions_AUPR, get_blacklist_mask, ChromosomeAUPRC


def run_benchmarking(args):
    """
    Benchmark a bigwig file of TF binding predictions against a gold standard file of predictions

    The inputs need to be in bigwig format to use this function. You can also provide a different blacklist to filter
    out regions that you do not want to include in your comparison. We use a np.mask to exclude these regions.

    Currently, benchmarking is set up for one chromosome at a time. The most time consuming step is importing and
    binning the input bigwig files to resolutions smaller than 100bp. We are also only benchmarking on whole chromosome
    at the moment so everything not in the blacklist will be considered a potential region.
    ___________________
    Workflow Overview

    1) Set up directories and names for this project
    2) Get the blacklist mask using the input blacklist and bin it at the same resolution as the predictions and GS
    3) Calculate the AUPR per chromosome

    :param args: output_directory, prefix, bin_size, prediction, gold_standard, chromosomes, agg_function,
    round_predictions

    :return: A tsv file of precision and recall with AUPR
    """
    # TODO set up benchmarking to include multiple chromosomes or only on specific regions.
    # Create the output directory
    output_dir = get_dir(args.output_directory)

    # Build the results filename
    results_filename = os.path.join(output_dir, args.prefix + "_" + str(args.bin_size) + "bp_PRC.tsv")

    logging.error(
        "Benchmarking" +
        "\n  Prediction file:" + args.prediction +
        "\n  Gold standard file: " + args.gold_standard +
        "\n  Bin size: " + str(args.bin_size) +
        "\n  Restricting to chromosomes: \n   - " + "\n   - ".join(args.chromosomes) +
        "\n  Output directory: " + output_dir +
        "\n  Output filename: " + results_filename + "\n"
    )

    # Calculate the AUPR using the prediction and gold standard
    ChromosomeAUPRC(args.prediction,
                    args.gold_standard,
                    args.blacklist,
                    args.chromosomes[0],
                    args.bin_size,
                    args.agg_function,
                    results_filename,
                    args.round_predictions)
