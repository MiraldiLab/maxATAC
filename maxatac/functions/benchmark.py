import logging
import os

from maxatac.utilities.benchmarking_tools import calculate_predictions_AUPR, get_blacklist_mask
from maxatac.utilities.system_tools import get_dir


def run_benchmarking(args):
    # Create the output directory
    output_dir = get_dir(args.output_directory)

    # Build the results filename
    results_filename = os.path.join(output_dir, args.prefix + "_" + str(args.bin_size) + "bp_PRC.tsv")

    logging.error(
        "Benchmarking" +
        "\n  Prediction file:" + args.prediction +
        "\n  Gold standard file: " + args.gold_standard +
        "\n  Bin size: " + str(args.bin_size) +
        "\n  Chromosomes: " + args.chromosomes[0] +
        "\n  Output directory: " + output_dir +
        "\n  Output filename: " + results_filename + "\n"
    )

    # Get the blacklist mask
    blacklist_mask = get_blacklist_mask(args.blacklist,
                                        bin_size=args.bin_size,
                                        chromosome=args.chromosomes[0])

    # Calculate the AUPR using the prediction and gold standard
    calculate_predictions_AUPR(args.prediction,
                               args.gold_standard,
                               args.bin_size,
                               args.chromosomes[0],
                               results_filename,
                               blacklist_mask,
                               args.agg_function,
                               args.round_predictions)

