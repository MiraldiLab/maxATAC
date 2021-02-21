import logging
import os

from maxatac.utilities.system_tools import get_dir, Mute

with Mute():
    from maxatac.utilities.genome_tools import build_chrom_sizes_dict
    from maxatac.utilities.constants import INPUT_CHANNELS, INPUT_LENGTH
    from maxatac.utilities.prediction_tools import write_predictions_to_bigwig, make_predictions, import_prediction_regions
    from maxatac.utilities.session import configure_session


def run_prediction(args):
    """
    Run prediction on compatible regions using a maxATAC model. The input could be any bed file that has the correct
    input parameters

    BED file requirements for prediction. You must have at least a 3 column file with chromosome, start,
    and stop coordinates. The interval distance has to be the same as the distance used to train the model. If you
    trained a model with a resolution 1024, you need to make sure your intervals are spaced 1024 bp apart for
    prediction with your model.

    Example input BED file for prediction:

    chr1 | 1000  | 2024
    ___________

    Workflow overview

    1) import_prediction_regions: Set up the output directory and filenames
    2) make_predictions: Import regions to predict on
    3) write_predictions_to_bigwig: Convert predictions to bigwig format and write results


    :param args : output_directory, prefix, signal, sequence, models, predict_chromosomes, minimum, threads, batch_size
    roi, chromosome_sizes, blacklist, average, round

    :return : A bigwig file of TF binding predictions
    """
    # Create the output directory set by the parser
    output_directory = get_dir(args.output_directory)

    # Output filename for the bigwig predictions file based on the output directory and the prefix. Add the bw extension
    outfile_name_bigwig = os.path.join(output_directory, args.prefix + ".bw")

    logging.error("Prediction Parameters \n" +
                  "Output filename: " + outfile_name_bigwig + "\n" +
                  "Target signal: " + args.signal + "\n" +
                  "Sequence data: " + args.sequence + "\n" +
                  "Models: \n   - " + "\n   - ".join(args.models) + "\n" +
                  "Chromosomes: " + str(args.predict_chromosomes) + "\n" +
                  "Minimum prediction value to be reported: " + str(args.minimum) + "\n" +
                  "Threads count: " + str(args.threads) + "\n" +
                  "Output directory: " + str(output_directory) + "\n" +
                  "Batch Size: " + str(args.batch_size) + "\n" +
                  "Output filename: " + outfile_name_bigwig + "\n"
                  )

    logging.error("Import BED file of regions for prediction")

    # TODO flag for whole genome vs chromosome vs regions
    # Import the regions for prediction.
    # The function build_chrom_sizes_dict is used to make sure regions fall within chromosome bounds.
    regions_pool = import_prediction_regions(bed_file=args.roi,
                                             region_length=INPUT_LENGTH,
                                             chromosomes=args.predict_chromosomes,
                                             chrom_sizes_dictionary=build_chrom_sizes_dict(args.predict_chromosomes,
                                                                                           args.chromosome_sizes
                                                                                           ),
                                             blacklist=args.blacklist
                                             )

    logging.error("Make predictions")

    configure_session(args.threads)

    # TODO Write the code so it can make prediction on multiple chromosomes and write them correctly to bigwig files.
    prediction_results = make_predictions(args.signal,
                                          args.sequence,
                                          args.average,
                                          args.models[0],
                                          regions_pool,
                                          args.batch_size,
                                          args.round,
                                          INPUT_CHANNELS,
                                          INPUT_LENGTH
                                          )

    logging.error("Write predictions to a bigwig file")

    # TODO before writing sort the data
    # TODO currently assumes that input is sorted by chr, start, stop in input
    # Write the predictions to a bigwig file
    write_predictions_to_bigwig(prediction_results,
                                output_filename=outfile_name_bigwig,
                                chrom_sizes_dictionary=build_chrom_sizes_dict(args.predict_chromosomes,
                                                                              args.chromosome_sizes
                                                                              ),
                                chromosomes=args.predict_chromosomes
                                )
