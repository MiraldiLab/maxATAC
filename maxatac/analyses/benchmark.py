import logging
import os
import timeit
from maxatac.utilities.system_tools import get_dir, Mute

with Mute():
    from maxatac.utilities.genome_tools import chromosome_blacklist_mask
    from maxatac.utilities.benchmarking_tools import ChromosomeAUPRC


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
    # Start Timer
    startTime = timeit.default_timer()

    # Create the output directory
    output_dir = get_dir(args.output_directory)

    # Build the results filename
    results_filename2 = os.path.join(output_dir, args.prefix + "_" + "r2_spearman_spearman.tsv")

    logging.error(
        "Benchmarking" +
        "\n  Prediction file:" + args.prediction +
        "\n  Gold standard file: " + args.gold_standard +
        "\n  Bin size: " + str(args.bin_size) +
        "\n  Restricting to chromosomes: \n   - " + "\n   - ".join(args.chromosomes) +
        "\n  Output directory: " + output_dir
    )

    # Calculate the AUPR using the prediction and gold standard
    for chromosome in args.chromosomes:
        logging.error("Benchmarking " + chromosome)
        # Build the results filename
        results_filename = os.path.join(output_dir,
                                        args.prefix + "_" + chromosome + "_" + str(args.bin_size) + "bp_PRC.tsv")

        ChromosomeAUPRC(args.prediction,
                        args.gold_standard,
                        args.blacklist,
                        chromosome,
                        args.bin_size,
                        args.agg_function,
                        results_filename,
                        args.round_predictions)

    # Measure End Time of Training
    stopTime = timeit.default_timer()
    totalTime = stopTime - startTime

    # Output running time in a nice format.
    mins, secs = divmod(totalTime, 60)
    hours, mins = divmod(mins, 60)

    logging.error("Total Benchmarking time: %d:%d:%d.\n" % (hours, mins, secs))