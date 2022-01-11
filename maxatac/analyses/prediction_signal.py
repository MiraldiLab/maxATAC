import logging
import os
import timeit
from maxatac.utilities.system_tools import get_dir, Mute

with Mute():
    from maxatac.utilities.genome_tools import chromosome_blacklist_mask
    from maxatac.utilities.prediction_signal_tools import Predicion_Signal


def run_prediction_signal(args):
    """
    Extracting the prediction signal for each specified bp window.

    :param args: predicted bigwig file, hg38.2bit, prefix bin_size, aggregation method (max), chromosomes, output
    :return: A tsv file of chr, start, stop with prediction signal

    The inputs need to be in bigwig format to use this function. You can also provide a different blacklist to filter
    out regions that you do not want to include in your comparison. We use a np.mask to exclude these regions.
    """
    # Start Timer
    startTime = timeit.default_timer()

    # Create the output directory
    output_dir = get_dir(args.output_directory)

    # Build the results filename
    logging.error(
        "Prediction_Signal" +
        "\n  Prediction file:" + args.prediction +
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

        Predicion_Signal(args.prediction,
                        args.sequence,
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

    logging.error("Total Prediction Signal calculation time: %d:%d:%d.\n" % (hours, mins, secs))